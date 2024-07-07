from torch.utils.data import Dataset
import json
from .squad_f1 import compute_exact, compute_f1
import numpy as np
from torch.utils.data import ConcatDataset
import csv
import torch
import os
from data_utils.mmlu import MMLUHelper
from data_utils.bbh import BBHHelper
import pickle
import random
from .utils import truncate_prefix

def fix_label_encoding(encoding, tokenizer):
    new_encoding = {}
    n = len(encoding['input_ids'])

    new_encoding['input_ids'] = torch.cat([torch.full(size=(n, 2), fill_value=tokenizer.bos_token_id),
                                           torch.LongTensor(encoding['input_ids'])], 1).cpu().numpy().tolist()
    new_encoding['attention_mask'] = torch.cat([torch.full(size=(n, 2), fill_value=1),
                                           torch.LongTensor(encoding['attention_mask'])], 1).cpu().numpy().tolist()
    return new_encoding

def load_bg_train_ds(config, tokenizer, ds_name, max_example, offset=0, use_cache=False, rand_sample=False, rand_seed=-1):
    path = os.path.join(config.pretrain_ds_dir, '{}.json'.format(ds_name))
    if use_cache:
        ds = P3Dataset.from_cache_and_file(path, ds_name, config, tokenizer, skip_encoding=False, max_example=max_example, offset=offset)
    elif rand_sample:
        ds = P3Dataset.from_file_sample(path, ds_name, config, tokenizer, skip_encoding=False,
                                        max_example=max_example, offset=offset, seed=rand_seed)
    else:
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer, skip_encoding=False, max_example=max_example,
                                 offset=offset)
    return ds

def load_bg_test_ds(config, tokenizer, ds_name, max_example, offset=0):
    path = os.path.join(config.pretrain_test_ds_dir, '{}.json'.format(ds_name))
    ds = P3Dataset.from_file(path, ds_name, config, tokenizer, skip_encoding=False, max_example=max_example, offset=offset)
    return ds

def load_ocl_dss(config, tokenizer):
    dss_names = config.ocl_tasks

    #else:
    train_dss, dev_dss, test_dss = {}, {}, {}
    for ds_name in dss_names:
        path = os.path.join(config.ocl_ds_dir, ds_name, 'upstream-5000.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer)
        train_dss[ds_name] = ds

        path = os.path.join(config.ocl_ds_dir, ds_name, 'validation-1000.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer)
        dev_dss[ds_name] = ds

        path = os.path.join(config.ocl_ds_dir, ds_name, 'test-1000.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer)
        test_dss[ds_name] = ds
    return train_dss, dev_dss, test_dss

def load_ocl_alt_ans_ds(config, task, tokenizer, name='test-1000-alt', skip_encoding=False):
    path = os.path.join(config.ocl_alt_ans_dir, task, '{}.json'.format(name))
    offset, max_instance = config.alt_p3.train_offset, config.alt_p3.train_max_instance
    ds = P3Dataset.from_alt_json(config, task, path, tokenizer, skip_encoding=skip_encoding, offset=offset, max_instance=max_instance)
    return ds

def load_ocl_alt_ans_ds_abl(config, task, tokenizer):
    offset, max_instance = config.alt_p3.train_offset, config.alt_p3.train_max_instance
    path = os.path.join(config.pretrain_ds_dir, '{}.json'.format(task))
    ds = P3Dataset.from_file(path,task, config, tokenizer, skip_encoding=False, max_example=max_instance,
                             offset=offset)
    return ds

def load_dir_tasks(dir_name):
    tasks = os.listdir(dir_name)
    return tasks

def load_ocl_ds_splits(config, task, tokenizer):
    path = os.path.join(config.ocl_ds_dir, task, 'upstream-5000.json'.format(task))
    train_ds = P3Dataset.from_file(path, task, config, tokenizer)

    path = os.path.join(config.ocl_ds_dir, task, 'validation-1000.json'.format(task))
    dev_ds = P3Dataset.from_file(path, task, config, tokenizer)

    path = os.path.join(config.ocl_ds_dir, task, 'test-1000.json'.format(task))
    test_ds = P3Dataset.from_file(path, task, config, tokenizer)

    return train_ds, dev_ds, test_ds


class P3Dataset(Dataset):
    def __init__(self, config, task_name, examples, tokenizer, indexes=None, skip_encoding=False, max_input_length=-1):
        self.input_texts, self.labels, self.indexes = [], [], indexes
        self.config = config
        self.tokenizer = tokenizer
        self.task_name = task_name

        if indexes is not None:
            assert len(indexes) == len(examples)

        for q, anss, meta in examples:
            assert len(anss) == 1
            self.input_texts.append(q)
            self.labels.append(anss[0])

        if config.is_llama:
            self.convert_to_llama_input_labels()

        self.skip_encoding = skip_encoding
        #print(task_name)
        if not skip_encoding:
            max_input_length = config.max_input_length if max_input_length == -1 else max_input_length

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if config.truncate_prefix:
                self.input_encoding = truncate_prefix(tokenizer, self.input_texts, max_len=max_input_length)
                self.label_encoding = truncate_prefix(tokenizer, self.labels, max_len=config.max_output_length)
            else:
                self.input_encoding = tokenizer.batch_encode_plus(self.input_texts, padding='max_length', truncation=True,
                                                                  max_length=max_input_length)
                self.label_encoding = tokenizer.batch_encode_plus(self.labels, padding='max_length', truncation=True,
                                                                  max_length=config.max_output_length,)
            if config.fix_label_bos:
                #raise ValueError('Fix bos is deprecated')
                print('Fixing bos encoding')
                self.label_encoding = fix_label_encoding(self.label_encoding, self.tokenizer)

        else:
            print('Skipping encoding')
            self.input_encoding, self.label_encoding = None, None

        if type(task_name) is not list:
            self.task_names = [task_name] * len(self.input_texts)
        else:
            self.task_names = task_name

    def convert_to_llama_input_labels(self):
        new_inputs, new_labels = [], []
        prompt_format = "[INST] <<SYS>>\nYou are an assistant that provide very succinct answers to questions.\n<</SYS>> " \
                        "{question} [\INST] The answer is"
        for text, answer in zip(self.input_texts, self.labels):
            new_input = prompt_format.format(question=text)
            new_label = new_input + ' ' + answer
            new_inputs.append(new_input)
            new_labels.append(new_label)
        self.input_texts, self.labels = new_inputs, new_labels

    @classmethod
    def from_cache_and_file(cls, path, task_name, config, tokenizer, skip_encoding=False, max_example=-1, offset=0):
        cache_dir = 'data/cache/{task_name}_{model}.pkl'.format(task_name=task_name, model=config.model_name.replace('/','+'))
        if os.path.isfile(cache_dir):
            print('Loading tokenizer cache at {}'.format(cache_dir))
            ds = cls.from_file(path, task_name, config, tokenizer, skip_encoding=True, max_example=-1, offset=0)

            with open(cache_dir,'rb') as f:
                obj = pickle.load(f)
            ds.input_encoding = obj['input_encoding']
            ds.label_encoding = obj['label_encoding']
            ds.skip_encoding = False
        else:
            print('Building tokenizer cache at {}'.format(cache_dir))
            ds = cls.from_file(path, task_name, config, tokenizer, skip_encoding=False, max_example=max_example,
                               offset=offset)
            with open(cache_dir, 'wb') as wf:
                pickle.dump({
                    'input_encoding': ds.input_encoding,
                    'label_encoding': ds.label_encoding
                }, wf)
        return ds


    @classmethod
    def from_file(cls, path, task_name, config, tokenizer, skip_encoding=False, max_example=-1, offset=0):
        if offset != 0:
            print('P3 from file: Offset is {}'.format(offset))

        with open(path) as f:
            examples = json.load(f)
            indexes = [_ for _ in range(len(examples))]

        if offset + max_example > len(examples):
            print('Not enough example for {}, {}'.format(task_name, len(examples)))

        if max_example != -1:
            examples = examples[offset:offset+max_example]
            indexes = indexes[offset:offset+max_example]
        else:
            examples = examples[offset:]
            indexes = indexes[offset:]

        ds = cls(config, task_name, examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_file_sample(cls, path, task_name, config, tokenizer, skip_encoding=False, max_example=-1, offset=0, seed=None):
        assert max_example != -1

        print('Sample P3 from file: Offset is {} / seed is {}'.format(offset, seed))

        with open(path) as f:
            examples = json.load(f)
            indexes = [_ for _ in range(len(examples))]

        if offset + max_example > len(examples):
            print('Not enough example for {}, {}'.format(task_name, len(examples)))

        examples, indexes = examples[offset:], indexes[offset:]
        samples = random.Random(seed).sample([(x,y) for x,y in zip(examples, indexes)], max_example)
        example_ss, index_ss = [x[0] for x in samples], [x[1] for x in samples]

        ds = cls(config, task_name, example_ss, tokenizer, indexes=index_ss, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_csv(cls, path, config, tokenizer, skip_encoding=False, max_example=-1):
        with open(path) as f:
            reader = csv.reader(f)
            rows = [_ for _ in reader]
            tasks = [_[-1] for _ in rows]

        examples = []
        for row in rows:
            examples.append([row[0], [row[1]], 'dummy'])
        indexes = [_ for _ in range(len(examples))]

        ds = cls(config, tasks, examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_rows(cls, rows, config, tokenizer, skip_encoding=False):
        tasks = [_[-1] for _ in rows]
        examples = []
        for row in rows:
            examples.append([row[0], [row[1]], 'dummy'])
        indexes = [_ for _ in range(len(examples))]

        ds = cls(config, tasks, examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_alt_json(cls, config, task_name, path, tokenizer, skip_encoding=False, offset=0, max_instance=-1):
        with open(path) as f:
            data = json.load(f)
        examples = [[dic['question'], [dic['alt_answer']], 'alt_answer'] for dic in data][offset:]
        if max_instance > 0:
            examples = examples[:max_instance]
        indexes = [_ for _ in range(len(examples))]
        ds = cls(config, task_name, examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

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
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type)
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
            all_task_names.extend(task_names)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, all_task_names, all_examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        for task in tasks:
            bbh_helper = BBHHelper(config, task)
            is_cot = config.bbh.is_cot
            examples = bbh_helper.create_examples(cot=is_cot)
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
            all_task_names.extend(task_names)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, all_task_names, all_examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding)
        return ds

    @classmethod
    def from_concat_example(cls, config, tokenizer, in_context_input, in_context_ans, raw_ds, skip_encoding=False,
                            concat_prompt=' '):
        all_examples = []
        tasks_names = [x['task_name'] for x in raw_ds]
        for item in raw_ds:
            concat_input = in_context_input + concat_prompt + in_context_ans + '\n\n' + item['original_input']
            ans = item['original_answers']
            all_examples.append([concat_input,[ans], 'dummy'])
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tasks_names, all_examples, tokenizer, indexes=indexes, skip_encoding=skip_encoding,
                 max_input_length=config.max_input_length*2)
        return ds

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, i):
        if self.skip_encoding:
            ret = {
                'original_input': self.input_texts[i],
                'original_answers': self.labels[i],
                'task_name': self.task_names[i],
                'idx': i
            }
        else:
            ret = {'input_ids': self.input_encoding['input_ids'][i],
                    'attention_mask': self.input_encoding['attention_mask'][i],
                    'labels': self.label_encoding['input_ids'][i], 'task_name': self.task_names[i],
                    'original_input': self.input_texts[i],
                    'original_answers': self.labels[i],
                    '_decoder_attention_mask': self.label_encoding['attention_mask'][i],
                    'idx': i}
        if self.indexes is not None:
            ret['index'] = self.indexes[i]
        return ret

    def compute_metrics(self, gts, preds):
        ems, f1s = [], []
        assert len(gts) == len(preds)
        for gt, pred in zip(gts, preds):
            ems.append(compute_exact(gt, pred))
            f1s.append(compute_f1(gt, pred))
        em_score, f1_score = np.mean(ems), np.mean(f1s)
        return {'EM': em_score, 'F1': f1_score}

    def compute_score_single(self, gt, pred):
        # EM, F1 - ordered like this
        em, f1 = compute_exact(gt, pred), compute_f1(gt, pred)
        return em, f1

    def get_score_func(self):
        def score_func(gt, pred):
            return self.compute_score_single(gt, pred)
        return score_func

    def group_score_by_task(self, scores):
        groups = {}
        assert len(self) == len(scores)
        for item, score in zip(self, scores):
            task = item['task_name']
            if task not in groups:
                groups[task] = []
            groups[task].append(score)
        avg_scores = {k: np.mean(v) for k,v in groups.items()}
        return avg_scores


class P3ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def compute_score_single(self, gt, pred):
        return self.datasets[0].compute_score_single(gt, pred)

    def compute_metrics(self, gts, preds):
        return self.datasets[0].compute_metrics(gts, preds)

    def get_score_func(self):
        return self.datasets[0].get_score_func()


def save_dataset(dataset, dst_file):
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        rows.append([item['original_input'], item['original_answers'], item['task_name']])
    with open(dst_file,'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)


