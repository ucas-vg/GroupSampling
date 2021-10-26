from __future__ import absolute_import
from collections import defaultdict

from copy import deepcopy
import random
import torch
from torch.utils.data.sampler import Sampler


class GroupSampler(Sampler):
    def __init__(self, dataset_labels, group_n=1, batch_size=None):
        label2data_idx = defaultdict(list)
        for i, (_, label, _) in enumerate(dataset_labels):
            label2data_idx[label].append(i)

        label2data = defaultdict(list)
        for label, data_idx in label2data_idx.items():
            if len(data_idx) > 1:
                label2data[label].extend(data_idx)
            else:
                label2data[-1].extend(data_idx)

        self.label2data_idx = label2data
        self.dataset_labels = dataset_labels
        self.group_n = group_n
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset_labels)

    def __iter__(self):
        data_idxes = []
        for label, data_idx in self.label2data_idx.items():
            if label != -1:
                data_idx = deepcopy(data_idx)
                random.shuffle(data_idx)
                data_idxes.extend([data_idx[i: i + self.group_n] for i in range(0, len(data_idx), self.group_n)])
                # data_idxes.append(data_idx)
        random.shuffle(data_idxes)
        ret = []
        for data_idx in data_idxes:
            ret.extend(data_idx)
        data_idx = deepcopy(self.label2data_idx[-1])
        random.shuffle(data_idx)
        ret.extend(data_idx)

        if self.batch_size is not None:
            batch_shuffle_ret = []
            tmp = [ret[i: i + self.batch_size] for i in range(0, len(ret), self.batch_size)]
            random.shuffle(tmp)
            for batch in tmp:
                batch_shuffle_ret.extend(batch)
            return iter(batch_shuffle_ret)
        else:
            return iter(ret)

    def __str__(self):
        return f"GroupSampler(num_instances={self.group_n}, batch_size={self.batch_size})"

    def __repr__(self):
        return self.__str__()

