from vllm import LLM, SamplingParams
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil
import time
import socket
from .data_utils.lm import SFTExampleOnlyDataset
from .utils.config import load_configs
import torch
import pickle
from peft import AutoPeftModelForCausalLM
from utils.analysis_tools import get_revision_name

hostname = socket.gethostname()
if hostname in ['ink-ruby', 'ink-nova']:
    model_dtype = 'half'
else:
    model_dtype = torch.bfloat16

def create_tmp_peft_merged_model(model_name, peft_model_dir, revision):
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir, revision=revision)
    #model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    #model.load_adapter(peft_model_dir)

    timestamp = time.time()
    tmp_dir = os.path.join(peft_model_dir, 'tmp_{}'.format(timestamp))
    os.makedirs(tmp_dir)
    model = model.merge_and_unload()
    model.save_pretrained(tmp_dir)
    return tmp_dir

def load_peft_ckpt(peft_model_dir, model_name, tokenizer_name, revision):
    tmp_dir = create_tmp_peft_merged_model(model_name, peft_model_dir, revision)
    llm = LLM(tmp_dir, trust_remote_code=True, dtype=model_dtype, tokenizer=tokenizer_name)
    shutil.rmtree(tmp_dir)
    return llm

def load_base_llm(model_name, tokenizer_name):
    llm = LLM(model_name, trust_remote_code=True, dtype=model_dtype, tokenizer=tokenizer_name)
    return llm

def load_ds(args, config, tokenizer, task_category, task_id=None):
    if task_category == 'mmlu':
        if args.ocl_split == 'test':
            print('Using test split')
            ds = SFTExampleOnlyDataset.from_mmlu([config.mmlu_tasks[task_id]], 'test', config)
        else:
            ds = SFTExampleOnlyDataset.from_mmlu([config.mmlu_tasks[task_id]],'val',config)
    elif task_category == 'tulu':
        ds = SFTExampleOnlyDataset.from_tulu(config)
    elif task_category == 'bbh':
        ds = SFTExampleOnlyDataset.from_bbh([config.bbh_tasks[task_id]],'val',config)
    elif task_category == 'dolma':
        ds = SFTExampleOnlyDataset.from_dolma(config)
    elif task_category == 'truthful_qa':
        ds = SFTExampleOnlyDataset.from_truthful_qa(config, [config.truthful_qa_tasks[task_id]])
    elif task_category == 'tulu_train':
        ds = SFTExampleOnlyDataset.from_tulu_train(config, [config.tulu_tasks[task_id]])
    else:
        raise NotImplementedError
    return ds

def run_stats_on_ds(args, config, llm, tokenizer, ds):
    results = {}
    if args.stat_ppl :
        sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
        chat_examples = ds.get_chat_or_raw_examples(include_gt=True, tokenizer=tokenizer)
        results['ppl'] = llm.generate(chat_examples, sampling_params)

    if args.stat_output:
        sampling_params = SamplingParams(temperature=0, max_tokens=512)
        chat_examples = ds.get_chat_or_raw_examples(include_gt=False, tokenizer=tokenizer)
        results['output'] = llm.generate(chat_examples, sampling_params)

    return results

def evaluate_llm_ocl_ds(args, config, tokenizer, llm):
    ocl_ds = load_ds(args, config, tokenizer, config.stat.ocl_task_category, config.stat.ocl_task_id)
    results = run_stats_on_ds(args, config, llm, tokenizer, ocl_ds)
    return results

def evaluate_llm_pt_ds(args, config, tokenizer, llm):
    pt_ds = load_ds(args, config, tokenizer, config.stat.pt_task_category, config.stat.pt_task_id)
    results = run_stats_on_ds(args, config, llm, tokenizer, pt_ds)
    return results

def save_sep_results(output_dir, name,results):
    for key in results:
        out_file = os.path.join(output_dir, f'{name}_{key}_results.pkl')
        with open(out_file, 'wb') as wf:
            pickle.dump(results[key], wf)

def format_name(args, ds_type):
    name = ds_type
    if args.ocl_split:
        name += '-{}'.format(args.ocl_split)
    if args.eval_base:
        name += '-base'
    return name

def main(args, config):

    if config.pt_revision:
        revision_name = get_revision_name(config, config.pt_revision)
    else:
        revision_name = None
    tokenizer_name = config.model_name if config.tokenizer_name is None else config.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              trust_remote_code=True)
    os.makedirs(config.output_dir, exist_ok=True)
    print('Output dir is {}'.format(config.output_dir))
    if args.eval_base:
        llm = load_base_llm(config.model_name, tokenizer_name)
    else:
        ckpt_dir = os.path.join(config.stat.task_model_dir)
        llm = load_peft_ckpt(ckpt_dir, config.model_name, tokenizer_name, revision_name)

    if not args.skip_eval_ocl_ds:
        ocl_results = evaluate_llm_ocl_ds(args, config, tokenizer, llm)
        save_sep_results(config.output_dir, format_name(args, 'ocl'), ocl_results)
    if not args.skip_eval_pt_ds:
        pt_results = evaluate_llm_pt_ds(args, config, tokenizer, llm)
        save_sep_results(config.output_dir, format_name(args, 'pt'), pt_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument('--eval_base', action='store_true')
    #parser.add_argument('--eval_task_model', action='store_true')

    parser.add_argument('--skip_eval_ocl_ds', action='store_true')
    parser.add_argument('--skip_eval_pt_ds', action='store_true')

    parser.add_argument('--stat_ppl', action='store_true')
    parser.add_argument('--stat_output', action='store_true')

    parser.add_argument('--ocl_split')
    args = parser.parse_args()

    config = load_configs(*args.config_files, templates=args.templates)
    main(args, config)