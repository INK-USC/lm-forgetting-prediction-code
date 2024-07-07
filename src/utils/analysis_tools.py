from .config import merge_config_into_args, load_configs
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from ..trainer.utils import trim_batch, DataCollatorWithPaddingStr
import os
from ..trainer.my_trainer import MyTrainer
from transformers import AdamW

import pickle
import numpy as np
import csv

import torch
import logging

import json
import hashlib
from tqdm import tqdm
from torch.nn.functional import softmax, cross_entropy

from ..data_utils.squad_f1 import compute_exact, compute_f1

from torch.utils.data import ConcatDataset

try:
    import peft
    from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
except ImportError:
    pass

logger = logging.getLogger()


def compute_score_single(gt, pred):
    # EM, F1 - ordered like this
    em, f1 = compute_exact(gt, pred), compute_f1(gt, pred)
    return em, f1

def run_evaluate(config, model, trainer, eval_ds_or_batches):
    if type(eval_ds_or_batches) is list:
        eval_dataloader = eval_ds_or_batches
    else:
        eval_dataloader = trainer.get_eval_dataloader_raw(eval_ds_or_batches, batch_size=config.per_device_eval_batch_size)
    all_preds, all_gts = [], []
    for batch in tqdm(eval_dataloader, desc='Evaluation', total=len(eval_dataloader)):
        input_ids, attn_masks = batch['input_ids'], batch['attention_mask']
        input_ids, attn_masks = trim_batch(input_ids, trainer.tokenizer.pad_token_id, attn_masks)
        input_ids, attn_masks = input_ids.cuda(), attn_masks.cuda()
        max_len = config.max_output_length
        if not config.is_seq2seq:
            max_len = config.max_input_length + config.max_output_length
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_masks,
            num_beams=config.num_beams,
            max_length=max_len,
            decoder_start_token_id=model.config.bos_token_id
        )
        #if not hasattr(logger, 'flg1'):
        #    logger.flg1 = 1
        #print('Decoder start token id is {}'.format(model.config.decoder_start_token_id))

        for b_idx in range(len(input_ids)):
            pred_ = trainer.tokenizer.decode(outputs[b_idx])
            pred = trainer.tokenizer.decode(outputs[b_idx], skip_special_tokens=True)
            if 'original_answers' in batch:
                gt = batch['original_answers'][b_idx]
            else:
                gt = batch['answer'][b_idx]
            all_preds.append(pred)
            all_gts.append(gt)
    return all_preds, all_gts

def run_evaluate_loss(config, model, trainer: MyTrainer, eval_ds):
    eval_dataloader = trainer.get_eval_dataloader_raw(eval_ds)
    all_losses = []
    for batch in eval_dataloader:
        avg_masked_loss = run_evaluate_loss_batch(config, model, trainer, batch)
        all_losses.extend(avg_masked_loss.detach().cpu().numpy().tolist())
    return all_losses

def load_peft_model(config, base_model):
    if 'BART0' in config.model_name:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=['q_proj', 'v_proj']
        )
    elif 't5' in config.model_name:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=['q', 'v']
        )
    elif 'OLMo' in config.model_name:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=['att_proj', 'ff_proj'],
            task_type="CAUSAL_LM",
        )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model

def run_evaluate_batch(config, model, trainer, batch):
    batch_preds, batch_gts = [], []
    input_ids, attn_masks = batch['input_ids'], batch['attention_mask']
    input_ids, attn_masks = trim_batch(input_ids, trainer.tokenizer.pad_token_id, attn_masks)
    input_ids, attn_masks = input_ids.cuda(), attn_masks.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attn_masks,
        num_beams=config.num_beams,
        max_length=config.max_output_length,
        decoder_start_token_id=model.config.bos_token_id
    )
    #if not hasattr(logger, 'flg2'):
    #    logger.flg2 = 1
    #print('Decoder start token id is {} in eval batch'.format(model.config.decoder_start_token_id))

    for b_idx in range(len(input_ids)):
        pred_ = trainer.tokenizer.decode(outputs[b_idx])
        pred = trainer.tokenizer.decode(outputs[b_idx], skip_special_tokens=True)
        gt = batch['original_answers'][b_idx]
        batch_preds.append(pred)
        batch_gts.append(gt)
    return batch_preds, batch_gts



@torch.no_grad()
def run_evaluate_loss_batch(config, model, trainer, batch):
    batch_ = trainer.clean_batch(batch)

    batch_['input_ids'], batch_['attention_mask'], batch_['labels'] = batch_['input_ids'].cuda(), \
                                                                      batch_['attention_mask'].cuda(), \
                                                                      batch_['labels'].cuda()
    batch_['output_hidden_states'] = True
    # get loss
    _, outputs = trainer.compute_loss(model, batch_, return_outputs=True)

    logits = outputs.logits
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    loss = cross_entropy(logits.view(-1, logits.size(2)), batch_['labels'].view(-1)).view(logits.size(0),-1) # [B,T]
    mask = batch['labels'].cuda() != -100
    masked_loss = torch.masked_fill(loss, ~mask, 0) # [B,T]
    lens = mask.long().sum(-1) # [B]
    avg_masked_loss = masked_loss.sum(-1) / (lens + 1e-10)
    return avg_masked_loss # [B]

def stat_loss_on_ds(config, model, trainer, pt_ds):
    all_losses = run_evaluate_loss(config, model, trainer, pt_ds)
    return all_losses

def stat_scores_on_ds(config, model, trainer, pt_ds, return_preds=False):
    all_preds, all_gts = run_evaluate(config, model, trainer, pt_ds)

    instance_met_scores = []
    for idx, (gt, pred) in enumerate(zip(all_gts, all_preds)):
        score = compute_score_single(gt, pred)
        instance_met_scores.append(score)
    if return_preds:
        return instance_met_scores, (all_preds, all_gts)
    return instance_met_scores

def stat_logits_on_ds_or_batches(config, ds_or_batches, model, tokenizer, trainer, topk=100):
    if type(ds_or_batches) is list:
        dataloader = ds_or_batches
    else:
        dataloader = trainer.get_eval_dataloader_raw(ds_or_batches)
    all_logits, all_labels = [], []
    all_probs = []
    all_losses = []

    all_logit_preds = []
    all_gts = []

    all_logit_pred_tokens = []

    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        _, logits = get_logit_batch(config, model, trainer, batch, return_loss=True) # [B,T,V]
        batch_size = logits.size(0)
        batch = trainer.clean_batch(batch)
        for b in range(batch_size):
            b_logits = logits[b]

            keep_mask = batch['labels'][b].ne(-100)

            b_logits = b_logits[keep_mask,:]
            b_labels = batch['labels'][b][keep_mask]

            topv, topi = b_logits.topk(topk)
            prob = softmax(b_logits, -1)
            topv_prob, _ = prob.topk(topk)

            all_logits.append([topv.cpu().detach().numpy(), topi.cpu().detach().numpy()])
            all_labels.append(b_labels.detach().numpy())
            all_probs.append(topv_prob.cpu().detach().numpy())

            loss = cross_entropy(b_logits, b_labels.cuda(), reduction='none')
            all_losses.append(loss.detach().cpu().numpy())

            all_logit_preds.append(tokenizer.decode(b_logits.max(-1)[1] , skip_special_tokens=False))
            all_gts.append(tokenizer.decode(b_labels, skip_special_tokens=False))

            all_logit_pred_tokens.append(b_logits.max(-1)[1].cpu().numpy())

    return all_logits, all_labels, all_probs, all_losses, all_logit_preds, all_logit_pred_tokens, all_gts

def stat_causal_lm_loss_on_ds_or_batches(config, ds_or_batches, model, tokenizer, trainer):

    max_step = 100000000
    if config.ocl_val_max_batch > 0:
        max_step = config.ocl_val_max_batch

    if type(ds_or_batches) is list:
        dataloader = ds_or_batches
    else:
        dataloader = trainer.get_eval_dataloader_raw(ds_or_batches)

    all_losses = []
    all_tasks = []

    model.eval()
    for idx, batch_ in tqdm(enumerate(dataloader), total=min(max_step, len(dataloader))):
        if idx == max_step:
            break
        batch = trainer.clean_batch(batch_)
        ref_loss, logits = get_logit_batch(config, model, trainer, batch, return_loss=True)  # [B,T,V]
        ce_loss = shift_cross_entropy(logits, batch['labels']) # [B,T]
        keep_mask = batch['labels'].ne(-100)
        seq_lens = keep_mask.float().sum(-1).cuda() # [B]
        ce_loss_per_seq = ce_loss.sum(-1) / (seq_lens + 1e-10) # [B]
        all_losses.extend(ce_loss_per_seq.cpu().numpy().tolist())
        all_tasks.extend(batch_['task_name'])

    return all_losses, all_tasks

def iscores2errors(iscores):
    # iscores: list of <em, f1>
    errs = []
    for idx, (em, f1) in enumerate(iscores):
        if em != 1:
            errs.append(idx)
    return errs

def load_all_records(exp='vanilla_bg100_lr1e-6', task='super_glue-copa', model_type='bart0'):
    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/aff_log.pkl', 'rb') as f:
        aff_obj = pickle.load(f)
        aff_score = np.mean([v for v in get_cnts(aff_obj).values()])

    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/aff_score_log.pkl', 'rb') as f:
        aff_score_obj = pickle.load(f)
        # aff_score = np.mean([v for v in get_cnts(aff_obj).values()])

    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/ocl_log.pkl', 'rb') as f:
        ocl_obj = pickle.load(f)
        succ_score = edit_success_rate(ocl_obj)

    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/ocl_error_ds.csv', 'r') as f:
        reader = csv.reader(f)
        ocl_rows = [_ for _ in reader]

    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/concat_pt_ds.csv', 'r') as f:
        reader = csv.reader(f)
        pt_rows = [_ for _ in reader]

    return {
        'aff_obj': aff_obj,
        'aff_score': aff_score,
        'aff_score_obj': aff_score_obj,
        'ocl_obj': ocl_obj,
        'succ_score': succ_score,
        'ocl_rows': ocl_rows,
        'pt_rows': pt_rows
    }


def load_loss_records(exp='vanilla_bg100_lr1e-6', task='super_glue-copa', model_type='bart0'):
    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/aff_loss_log.pkl', 'rb') as f:
        aff_loss_obj = pickle.load(f)
    return aff_loss_obj


def get_cnts(dic):
    return {k: len(v) for k, v in dic.items()}


def edit_success_rate(dic):
    s, c = 0, 0
    for idx, entry in dic.items():
        if entry['after'][0] == 1:
            c += 1
        s += 1
    return c / s


def load_features(exp='vanilla_bg100_lr1e-6', task='super_glue-cb', common_task='super_glue-cb', model_type='bart0'):
    with open(f'runs/instance-p3-{model_type}/{exp}/{common_task}/features/pt.pkl', 'rb') as f:
        pt_feats = pickle.load(f)

    with open(f'runs/instance-p3-{model_type}/{exp}/{task}/features/ocl_errors.pkl', 'rb') as f:
        ocl_error_feats = pickle.load(f)

    return pt_feats, ocl_error_feats


def get_new_errors(before, after):
    before = set(before)
    return [x for x in after if x not in before]


def initialize(config_files, templates):
    config = load_configs(*config_files, templates=templates)
    if config.is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)

    if config.peft == 'lora':
        model = load_peft_model(config=config, base_model=model)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    data_collator = DataCollatorWithPaddingStr(tokenizer)

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config
    )

    return config, model, tokenizer, trainer, data_collator

def initialize_no_model(config_files, templates):
    config = load_configs(*config_files, templates=templates)

    #model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_collator = DataCollatorWithPaddingStr(tokenizer)

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    return config, tokenizer, data_collator

def save_model_state(model, optimizer):
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}
    return model_state, optim_state


def reset(model, optimizer, model_state, optim_state):
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)


def make_batch(ocl_example, collator):
    batch = collator([ocl_example])
    return batch

def update_on_example(config, model, trainer, batch, optimizer, eval_mode):
    #config.ocl_steps = 30
    #config.max_grad_norm = -1
    logger.info(f'Doing {config.ocl_steps} step of updates')
    trainer.nstep_model_update(model, batch, optimizer, n_step=config.ocl_steps, eval_mode=eval_mode)


def get_raw_gradients(config, model, batch, trainer, clip=False):
    all_grads = {}
    model.zero_grad()
    batch_ = trainer.clean_batch(batch)
    loss = trainer.training_step_eval_mode(model, batch_)

    if clip:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )


    for n, p in model.named_parameters():
        if p.grad is not None:
            all_grads[n] = p.grad.detach() #.cpu().detach()
    return all_grads#, loss


def get_base_optimizer(config, model, training_args):
    optim_params = [x for x in model.parameters() if x.requires_grad]

    if config.optimizer_type == 'AdamW':
        optimizer = AdamW(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                          betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_epsilon)
    return optimizer

def get_raw_gradients_and_apply(model, batch, trainer, optimizer):
    all_grads = {}
    model.zero_grad()
    batch_ = trainer.clean_batch(batch)
    loss = trainer.training_step(model, batch_)
    for n, p in model.named_parameters():
        if p.grad is not None:
            all_grads[n] = p.grad.detach()

    model.zero_grad()
    return all_grads


def get_param_differences(model_state1, model_state2):
    all_diffs = {}
    for k in model_state1.keys():
        a, b = model_state1[k], model_state2[k]
        all_diffs[k] = b - a
        #all_diffs[k] = all_diffs[k]
    return all_diffs


def get_prod_and_dist(grads1, grads2):
    prod = {}
    dist = {}
    for k in grads1.keys():
        if k in grads2:
            a, b = grads1[k].cuda(), grads2[k].cuda()
            prod[k] = torch.dot(a.view(-1), b.view(-1)).cpu().detach()
            prod[k] = prod[k].item()

            if a.norm() != 0 and b.norm() != 0:
                dist[k] = prod[k] / (a.norm() * b.norm())
                dist[k] = dist[k].cpu().detach()
                dist[k] = dist[k].item()
            else:
                dist[k] = 0
    return prod, dist


@torch.no_grad()
def get_logit_batch(config, model, trainer, batch, return_loss=False):
    batch_preds, batch_gts = [], []
    batch_ = trainer.clean_batch(batch)

    batch_['input_ids'], batch_['attention_mask'], batch_['labels'] = batch_['input_ids'].cuda(), \
                                                                      batch_['attention_mask'].cuda(), \
                                                                      batch_['labels'].cuda()
    batch_['output_hidden_states'] = True
    # get loss
    loss, outputs = trainer.compute_loss(model, batch_, return_outputs=True)

    logits = outputs.logits
    if return_loss:
        return loss, logits
    else:
        return logits

def shift_cross_entropy(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits_ = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_ = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels_ = shift_labels_.to(shift_logits.device)
    loss = loss_fct(shift_logits_, shift_labels_)
    loss = loss.view(shift_logits.size(0), shift_logits.size(1))
    return loss

@torch.no_grad()
def get_loss_batch(config, model, trainer, batch):
    batch_preds, batch_gts = [], []
    batch_ = trainer.clean_batch(batch)

    batch_['input_ids'], batch_['attention_mask'], batch_['labels'] = batch_['input_ids'].cuda(), \
                                                                      batch_['attention_mask'].cuda(), \
                                                                      batch_['labels'].cuda()
    # get loss
    loss, outputs = trainer.compute_loss(model, batch_)
    return loss

def get_p3_hash(example):
    s = json.dumps(example, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def stat_replayed_vs_others_scores(replayed_idxs, pt_iscores):
    replayed_iscores = [x for i,x in enumerate(pt_iscores) if i in replayed_idxs]
    others_iscores = [x for i,x in enumerate(pt_iscores) if i not in replayed_idxs]

    replayed_errors = iscores2errors(replayed_iscores)
    others_errors = iscores2errors(others_iscores)

    return replayed_errors, others_errors

def create_past_model(config):
    past_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    return past_model

def create_extended_dataset(ds, repeat_time):
    dss = [ds for _ in range(repeat_time)]
    concat_ds = ConcatDataset(dss)
    return concat_ds

def get_revision_name(config, revision_idx):
    with open(config.pt_revision_list) as f:
        lines = f.readlines()
    revision = lines[-revision_idx].strip()
    return revision

def get_reference_train_step(config, task_idx, multiplier=1000):
    base_dir = os.path.join(config.ref_ocl_step_dir, 'task_{}'.format(task_idx))
    files = os.listdir(base_dir)
    losses = None
    for file in files:
        if file.endswith('_loss.json'):
            with open(os.path.join(base_dir, file)) as f:
                losses = json.load(f)['ocl_loss']
    min_idx = np.argmin(losses)
    return (min_idx + 1) * multiplier
