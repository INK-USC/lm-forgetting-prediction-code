import random as random_
import pickle
import numpy as np
import torch
from scipy.special import softmax

np_rng = np.random.default_rng()
class DatasetMemory:
    def __init__(self, ds, collator, random_seed=0):
        self.ds = ds
        self.collator = collator
        self.random = random_.Random(random_seed)

    def random_sample(self, k):
        if k < len(self.ds):
            sample_idxs = self.random.sample(range(len(self.ds)), k)
        else:
            sample_idxs = [_ for _ in range(len(self.ds))]
        examples = []
        tasks = []
        for idx in sample_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sample_idxs

    def random_sample_from_indices_with_filling(self, k, indices, replayed_idxs=None, no_repeat=False):
        if no_repeat:
            indices = [x for x in indices if x not in replayed_idxs]
        if k > len(indices):
            indices = indices + self.random.sample(range(len(self.ds)), k - len(indices))

        sample_idxs = self.random.sample(indices, k)
        examples = []
        tasks = []
        for idx in sample_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sample_idxs

    def weight_random_sampling(self, k, pred_forgets, weight_temp):
        weight = softmax(pred_forgets / weight_temp)
        sampled_idxs = np_rng.choice(len(self.ds), size=k, replace=False, p=weight)

        examples = []
        tasks = []
        for idx in sampled_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sampled_idxs
