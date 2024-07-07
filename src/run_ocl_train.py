from .trainer.my_trainer import MyTrainer
from .trainer.utils import trim_batch, DataCollatorWithPaddingStr, DataCollatorWithPaddingStrForLM, DataCollatorMaskedStrForLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from .utils.config import merge_config_into_args, load_configs
from .data_utils.lm import SFTDataset
from .data_utils import load_ocl_ds_by_task_id
import os
import logging
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD
import torch
import argparse
from tqdm import tqdm
from .utils.analysis_tools import \
    create_past_model, load_peft_model, \
    stat_causal_lm_loss_on_ds_or_batches, get_revision_name, get_reference_train_step
from .trainer.memory_utils import DatasetMemory
import numpy as np
import json

logger = logging.getLogger('main')

def get_memory_collator(config, tokenizer):
    if config.is_mem_chat_format:
        data_collator = DataCollatorMaskedStrForLM(tokenizer, max_length=config.max_input_length,
                                                       ans_start_patt="<|assistant|>")
    else:
        data_collator = DataCollatorWithPaddingStrForLM(tokenizer, max_length=config.max_input_length)
    return data_collator

def run_pipeline_stat_errors_in_stream(args):
    config = load_configs(*args.config_files, templates=args.templates)

    print(config.output_dir)

    model_kwargs = {}
    if config.pt_revision:
        revision_name = get_revision_name(config, config.pt_revision)
        model_kwargs['revision'] = revision_name
        print('Model revision is {}'.format(model_kwargs['revision']))
    if config.is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, trust_remote_code=True, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, **model_kwargs)

    if config.peft == 'lora':
        print('Creating peft model')
        model = load_peft_model(config=config, base_model=model)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if config.load_ckpt:
        print('Loading model from {}'.format(config.load_ckpt))
        state_dict = torch.load(config.load_ckpt)
        # state_dict = {k[len('model.'):]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name if config.tokenizer_name is None else config.tokenizer_name,
                                              trust_remote_code=True)

    if config.is_seq2seq:
        data_collator = DataCollatorWithPaddingStr(tokenizer)
    elif config.is_lm_sft:
        data_collator = DataCollatorMaskedStrForLM(tokenizer, max_length=config.max_input_length, ans_start_patt="<|assistant|>")
    else:
        data_collator = DataCollatorWithPaddingStrForLM(tokenizer, max_length=config.max_input_length)

    training_args = TrainingArguments(output_dir=config.output_dir)

    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config,
        past_model_creator=create_past_model
    )

    optim_params = [x for x in model.parameters() if x.requires_grad]
    if config.optimizer_type == 'AdamW':
        optimizer = AdamW(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                    betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_epsilon)
    elif config.optimizer_type == 'SGD':
        optimizer = SGD(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        print('Using SGD, configs: {}'.format(optimizer))
    else:
        raise NotImplementedError

    ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, config.ocl.task_category, config.ocl.task_id)
    print('Working on task {}:{} / saving at {}'.format(config.ocl.task_category, task, config.output_dir))

    ocl_train_loader = DataLoader(ocl_train_ds, config.stream.bs, shuffle=False, collate_fn=data_collator)

    all_val_losses = []

    if not args.skip_before_eval:
        before_val_losses, _ = stat_causal_lm_loss_on_ds_or_batches(config, ocl_eval_ds, model, tokenizer, trainer)
        before_val_loss = np.mean(before_val_losses)

        logger.info('Before training - val loss {} on task {}'.format(before_val_loss, task))
    else:
        before_val_loss = None

    step = 0
    min_loss = 1e10
    should_stop = False
    epoch_num = 0
    ocl_steps = config.ocl_steps
    do_replay = config.cl_method in ['er','mir_pred']
    replay_every = config.replay_freq

    if do_replay:
        replay_ds = SFTDataset.from_auto(config.pt_ds_name, tasks=None,
                                            split=None, config=config, tokenizer=tokenizer, skip_encoding=False)
        mem_collator = get_memory_collator(config, tokenizer)
        ds_memory = DatasetMemory(replay_ds, mem_collator)
    else:
        ds_memory = None

    if config.use_ref_ocl_train_step:
        ocl_steps = get_reference_train_step(config, config.ocl.task_id)
        print('Using reference ocl train steps {}'.format(ocl_steps))
    best_state = None

    pbar = tqdm(total=ocl_steps, desc='Training step')
    while True:
        for batch in ocl_train_loader:
            if step == ocl_steps:
                should_stop = True

            if should_stop:
                break

            pbar.update(n=1)

            if do_replay and step % replay_every == 0:
                replay_batch = trainer.get_replay_batch(ds_memory, config.ocl.task_id)
            else:
                replay_batch = None

            loss, step, loss_mem = trainer.model_update_accum_grad(model, batch, step, optimizer, mem_batch=replay_batch)

            if step % config.ocl_val_step == 0:
                val_losses, _ = stat_causal_lm_loss_on_ds_or_batches(config, ocl_eval_ds, model, tokenizer, trainer)
                val_loss = np.mean(val_losses)
                logger.info('Val loss {} on task {}'.format(val_loss, task))
                all_val_losses.append(val_loss)
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_state = {k: v.cpu() for k,v in model.state_dict().items()}
        epoch_num += 1
        if should_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.save_pretrained(os.path.join(config.output_dir, 'model_save'))

    with open(os.path.join(config.output_dir, '{}_ocl_loss.json'.format(task)),'w') as wf:
        json.dump({'ocl_loss': all_val_losses}, wf)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--ocl_task")
    parser.add_argument("--skip_before_eval", action='store_true')

    parser.add_argument("--n_gpu", default=1, type=int)

    args = parser.parse_args(argv)
    run_pipeline_stat_errors_in_stream(args)

if __name__ == '__main__':
    main()
