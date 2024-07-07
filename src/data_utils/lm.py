from torch.utils.data import Dataset
import numpy as np
import os
from .utils import truncate_prefix
from .mmlu import MMLUHelper
from .bbh import BBHHelper
from .utils import apply_chat_template, apply_chat_template_for_generation
import json
import pickle
import datasets


class SFTDataset(Dataset):
    def __init__(self, config, tokenier, input_texts, task_names, indexes):
        self.config = config
        self.tokenizer = tokenier
        self.input_texts = input_texts
        self.input_encoding = truncate_prefix(tokenier, input_texts, self.config.max_input_length)
        self.indexes = indexes
        self.task_names = task_names

    @classmethod
    def from_mmlu(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = BBHHelper(config, task)
            cot = config.bbh.is_cot
            examples = mmlu_helper.create_examples(split, cot=cot,  example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_flan_by_task(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        if split == 'dev':
            split_name = 'validation'
        else:
            split_name = split

        for task in tasks:
            with open(os.path.join(config.flan_by_task_dir, '{}_{}.json'.format(task, split_name))) as f:
                data = json.load(f)
            examples = [[x['inputs'], x['targets'], x['task']] for x in data]
            all_examples.extend(examples)
            all_task_names.extend([example[-1] for example in examples])

        if getattr(config, 'max_flan_example', -1) > 0:
            print('Max flan example is {} for FPD'.format(config.max_flan_example))

            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_tulu_train(cls, tasks, split, config, tokenizer, skip_encoding=False):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])

        all_task_names = [_[-1] for _ in all_examples]
        if getattr(config, 'max_tulu_train_example', -1) > 0:
            print('Max tulu_train example is {}'.format(config.max_tulu_train_example))
            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]

        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_tulu(cls, tasks, split, config, tokenizer, skip_encoding=False):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        all_examples = [
            [example['messages'][0]['content'],
             example['messages'][1]['content'],
             example['dataset'].split('.')[0]] for example in raw_examples
        ]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        all_task_names = [_[-1] for _ in all_examples]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_truthful_qa(cls, tasks, split, config, tokenizer, skip_encoding=False):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        all_task_names = [_[-1] for _ in all_examples]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_dolma_sample(cls, tasks, split, config, tokenizer, skip_encoding=False):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        input_texts = [x[0] for x in all_examples]
        all_task_names = [x[-1] for x in all_examples]
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        elif ds_category == 'tulu':
            return cls.from_tulu(**kwargs)
        elif ds_category == 'truthful_qa':
            return cls.from_truthful_qa(**kwargs)
        elif ds_category == 'tulu_train':
            return cls.from_tulu_train(**kwargs)
        elif ds_category == 'dolma_sample':
            return cls.from_dolma_sample(**kwargs)
        elif ds_category == 'flan':
            return cls.from_flan_by_task(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        input_ids = self.input_encoding['input_ids'][idx]
        task_name = self.task_names[idx]

        example = {
            'input_ids': input_ids,
            'task_name': task_name
        }
        return example

    def __len__(self):
        return len(self.input_encoding['input_ids'])


class SFTExampleOnlyDataset(Dataset):
    def __init__(self, examples, is_lm=False):
        self.examples = examples
        self.is_lm = is_lm

    @classmethod
    def from_mmlu(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            bbh_helper = BBHHelper(config, task)

            cot = config.bbh.is_cot
            examples = bbh_helper.create_examples(split,  cot=cot, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu(cls, config):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        all_examples = [
            [example['messages'][0]['content'],
             example['messages'][1]['content'],
             example['dataset'].split('.')[0]] for example in raw_examples
        ]
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_dolma(cls, config):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_truthful_qa(cls, config, tasks):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu_train(cls, config, tasks):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])
        ds = cls(all_examples)
        return ds


    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def get_chat_or_raw_examples(self, include_gt, tokenizer):
        if self.is_lm:
            input_texts = [x[0] for x in self.examples]
        else:
            if include_gt:
                input_texts = apply_chat_template(self.examples, tokenizer)
            else:
                input_texts = apply_chat_template_for_generation(self.examples, tokenizer)
        return input_texts

    def get_gt_answers(self):
        return [example[1] for example in self.examples]