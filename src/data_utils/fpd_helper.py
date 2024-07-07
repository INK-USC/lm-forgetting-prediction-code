from torch.utils.data import Subset, ConcatDataset
import torch
from .lm import SFTDataset
import pickle
import os
import random as random_
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from transformers import DataCollatorWithPadding

import logging
logger = logging.getLogger('fpd_helper')

MAX_N_PER_TASK = 10000

random = random_.Random(0)

def make_batch(ocl_example, collator):
    batch = collator([ocl_example])
    return batch

class FpdP3Helper:
    def __init__(self, config, tokenizer, data_collator, ocl_task):
        self.config = config
        self.tokenizer = tokenizer
        self.ocl_task = ocl_task
        self.collator = data_collator

        self._global_sample_state = 0

        print('Load fpd pre-splitted files - task level')
        self.train_ocl_dss, self.test_ocl_dss, self.pt_ds, self.train_mat, self.test_mat = self.prepare_from_split_sep_tasks()
        self.dev_ocl_dss, self.dev_mat = self.test_ocl_dss, self.test_mat
        self.train_pt_ds = self.dev_pt_ds = self.test_pt_ds = self.pt_ds
        self.train_ocl_error_ds = ConcatDataset(self.train_ocl_dss)
        self.test_ocl_error_ds = ConcatDataset(self.test_ocl_dss)
        self.dev_ocl_error_ds = ConcatDataset(self.dev_ocl_dss)


    def prepare_from_split_sep_tasks(self):
        def get_sft_dss(task_infos):
            dss = []
            for task_info in task_infos:
                task_cat, task_name, task_split = task_info['cat'], task_info['name'], task_info['split']
                ds = SFTDataset.from_auto(task_cat, tasks=[task_name], split=task_split,
                                          config=self.config,
                                          tokenizer=self.tokenizer)
                dss.append(ds)
            return dss

        with open(self.config.fpd.fpd_split_file, 'rb') as f:
            fpd_split = pickle.load(f)

        train_ocl_dss = get_sft_dss(fpd_split['train_ocl_task_info'])
        test_ocl_dss = get_sft_dss(fpd_split['test_ocl_task_info'])
        pt_ds_full = SFTDataset.from_auto(fpd_split['pt_task_info']['cat'], tasks=fpd_split['pt_task_info']['names'], split=fpd_split['pt_task_info']['split'],
                                     config=self.config, tokenizer=self.tokenizer)
        if 'ss_idxs' in fpd_split['pt_task_info']:
            pt_ds_ss_idxs = fpd_split['pt_task_info']['ss_idxs']
            pt_ds = Subset(pt_ds_full, pt_ds_ss_idxs)
        else:
            pt_ds = pt_ds_full

        train_mat, test_mat = fpd_split['train_mat'], fpd_split['test_mat']
        train_mat = np.nan_to_num(train_mat)
        test_mat = np.nan_to_num(test_mat)
        return train_ocl_dss, test_ocl_dss, pt_ds, train_mat, test_mat

    def mat_to_bin_fgt_list(self, mat):
        forgets = {}
        arr = np.arange(mat.shape[1])
        for idx in range(mat.shape[0]):
            forgets[idx] = arr[mat[idx] > 1e-10].tolist()
        return forgets

    def prepare_gt_forgets(self, aff_log_path, ocl_log_path):
        gt_forgets = {}
        with open(aff_log_path,'rb') as f:
            aff_log = pickle.load(f)
        with open(ocl_log_path,'rb') as f:
            ocl_log = pickle.load(f)
        base_errors = aff_log['before']
        ocl_idxs = sorted([x for x in ocl_log.keys()])

        for ocl_error_idx, ocl_idx in enumerate(ocl_idxs):
            gt_forgets[ocl_error_idx] = sorted([x for x in aff_log[ocl_idx] if x not in base_errors])
        return gt_forgets, base_errors

    def get_pt_dataloader(self, split, batch_size):
        ds = getattr(self, f'{split}_pt_ds')
        loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def get_pt_ds(self, split):
        ds = getattr(self, f'{split}_pt_ds')
        return ds

    def get_ocl_dataloader(self, split, batch_size):
        ds = getattr(self, f'{split}_ocl_error_ds')
        loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def get_all_gt_forgets(self, split):
        return getattr(self, f'{split}_gt_forgets')

    def sample_episode_task_level_balanced(self, split, bs):
        ocl_dss, pt_ds = getattr(self, '{}_ocl_dss'.format(split)), self.pt_ds
        fgt_mat = getattr(self, '{}_mat'.format(split))
        ocl_ds_num, pt_ex_num = len(ocl_dss), len(pt_ds)

        examples = []
        for b in range(bs):
            ocl_ds_idx = random.choice(range(ocl_ds_num))
            ocl_ds = ocl_dss[ocl_ds_idx]
            ocl_idx = random.choice(range(len(ocl_ds)))
            ocl_example = ocl_ds[ocl_idx]

            forgotten_idx = np.arange(len(pt_ds))[fgt_mat[ocl_ds_idx] > 0]
            non_forgotten_idx = np.arange(len(pt_ds))[fgt_mat[ocl_ds_idx] <= 0]

            if len(forgotten_idx) > 0:
                pos_idx = random.choice(forgotten_idx)
                pos_example = pt_ds[pos_idx]
                label = fgt_mat[ocl_ds_idx, pos_idx]
                if self.config.fpd.binarilize_labels:
                    label = 1 if label > 0 else 0
                examples.append({
                    'input_ids': pos_example['input_ids'],
                    'input_ids_ocl': ocl_example['input_ids'],
                    'input_ids_pt': pos_example['input_ids'],
                    'forget_label': label,
                    'pt_idx': pos_idx,
                    'ocl_ds_idx': ocl_ds_idx,
                    'ocl_ex_idx': ocl_idx
                })
            if len(non_forgotten_idx) > 0:
                neg_idx = random.choice(non_forgotten_idx)
                neg_example = pt_ds[neg_idx]
                label = fgt_mat[ocl_ds_idx, neg_idx]
                if self.config.fpd.binarilize_labels:
                    label = 1 if label > 0 else 0
                examples.append({
                    'input_ids': neg_example['input_ids'],
                    'input_ids_ocl': ocl_example['input_ids'],
                    'input_ids_pt': neg_example['input_ids'],
                    'forget_label': label,
                    'pt_idx': neg_idx,
                    'ocl_ds_idx': ocl_ds_idx,
                    'ocl_ex_idx': ocl_idx
                })
        batch = self.collator(examples)
        return batch

    def sample_episode_task_level(self, split, bs):
        ocl_dss, pt_ds = getattr(self, '{}_ocl_dss'.format(split)), self.pt_ds
        fgt_mat = getattr(self, '{}_mat'.format(split))
        ocl_ds_num, pt_ex_num = len(ocl_dss), len(pt_ds)

        examples = []
        for b in range(bs):
            pt_idx, ocl_ds_idx = random.choice(range(pt_ex_num)), random.choice(range(ocl_ds_num))
            ocl_ds = ocl_dss[ocl_ds_idx]
            ocl_idx = random.choice(range(len(ocl_ds)))

            pt_example, ocl_example = pt_ds[pt_idx], ocl_ds[ocl_idx]
            label = fgt_mat[ocl_ds_idx, pt_idx]
            if self.config.fpd.binarilize_labels:
                label = 1 if label > 0 else 0

            example = {
                'input_ids': pt_example['input_ids'],
                'input_ids_ocl': ocl_example['input_ids'],
                'input_ids_pt': pt_example['input_ids'],
                'forget_label': label,
                'pt_idx': pt_idx,
                'ocl_ds_idx': ocl_ds_idx,
                'ocl_ex_idx': ocl_idx
            }
            examples.append(example)
        batch = self.collator(examples)
        return batch

    def get_ocl_dataloader_concat(self, split, batch_size):
        ocl_dss = getattr(self, '{}_ocl_dss'.format(split))
        concat_ds = ConcatDataset(ocl_dss)
        loader = DataLoader(concat_ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def expand_scores(self, ocl_dss, scores):
        score_expand = []
        for ocl_idx in range(len(ocl_dss)):
            score_expand.extend([scores[ocl_idx]] * len(ocl_dss[ocl_idx]))
        score_expand = np.stack(score_expand)
        score_expand = torch.from_numpy(score_expand)
        return score_expand

    def get_ground_truth_mat(self, split):
        ocl_error_dss = getattr(self, f'{split}_ocl_dss')
        pt_ds = getattr(self, f'{split}_pt_ds')
        scores = torch.from_numpy(getattr(self, f'{split}_mat'))
        label_mat = self.expand_scores(ocl_error_dss, scores)
        bin_label_mat = torch.where(label_mat > 0, 1, 0)

        return label_mat, bin_label_mat

    def evaluate_metrics(self, fgt_label_grid, preds_grid):
        f1s = []
        ps, rs = [], []
        for ocl_error_idx in range(fgt_label_grid.size(0)):
            valid_label = fgt_label_grid[ocl_error_idx].detach().cpu().numpy()
            valid_pred = preds_grid[ocl_error_idx].detach().cpu().numpy()

            f1 = f1_score(valid_label, valid_pred)
            f1s.append(f1)
            p, r = precision_score(valid_label, valid_pred), recall_score(valid_label, valid_pred)
            ps.append(p)
            rs.append(r)
        ret = {
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'p_mean': np.mean(ps),
            'p_std': np.std(ps),
            'r_mean': np.mean(rs),
            'r_std': np.std(rs),
            'task': self.ocl_task
        }
        return ret

    def save_raw_scores(self, rep_prods, output_dir, split):
        if torch.is_tensor(rep_prods):
            rep_prods = rep_prods.cpu().numpy()
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'rep_prods_{split}.npy'), rep_prods)

    def predict_thres_based(self, freqs, perc):
        freq_vs = sorted([_ for _ in freqs.values()])
        thres = np.percentile(freq_vs, perc)
        preds = [k for k, v in freqs.items() if v > thres]
        # print(len(preds), thres)
        return preds


class DataCollatorWithPaddingStrForFpd(DataCollatorWithPadding):
    def pad_logits_or_idxs(self, tensor_list):
        max_len = max([len(x) for x in tensor_list])
        dim_size = tensor_list[0].size(-1)
        out = torch.zeros(len(tensor_list), max_len, dim_size, dtype=tensor_list[0].dtype)
        for i, tensor in enumerate(tensor_list):
            out[i, :len(tensor)] = tensor
        return out

    def __call__(self, features):
        features_non_str = []
        features_str = []

        special_feat_names = ['pt_logits_ss', 'pt_logits_idxs', 'ocl_update_logits','pt_logits_after_ss','pt_logits_after_idxs']
        special_feats = {k: [] for k in special_feat_names}

        for feature in features:
            dic, dic2 = {}, {}
            for k, v in feature.items():
                if k in special_feat_names:
                    special_feats[k].append(v)
                elif type(v) is not str:
                    dic[k] = v
                else:
                    dic2[k] = v
            features_non_str.append(dic)
            features_str.append(dic2)

        batch = self.tokenizer.pad(
            features_non_str,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        for k, vs in special_feats.items():
            if vs:
                out = self.pad_logits_or_idxs(vs)
                batch[k] = out

        for dic in features_str:
            for k, v in dic.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        return batch

