from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


# When num > num_instances, take (num - num % num_instances)
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            pid = data[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# When num > num_instances, only take num_instances
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances, epoch=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        self.epoch = epoch

    def __iter__(self):
#         indices = torch.randperm(self.num_identities)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             replace = False if len(t) >= self.num_instances else True
#             if len(t) > 1:
#                 t = np.random.choice(t, size=self.num_instances, replace=replace)
#             ret.extend(t)
#         # random.shuffle(ret)
#         return iter(ret)
        #
        # # add by hxm
        # num_batch = len(ret) // self.batch_size
        # new_ret = np.array(ret[:(num_batch * self.batch_size)]).reshape(self.batch_size, num_batch)
        # np.random.shuffle(new_ret)
        # return iter(new_ret.reshape(-1).tolist())

        indices = torch.randperm(self.num_identities)
        ret = []
        cluster = []
        uncluster = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            if len(t) > 1:
                t = np.random.choice(t, size=self.num_instances, replace=replace)
                cluster.extend(t)
            else:
                uncluster.extend(t)
        ret = cluster + uncluster  # [:(len(uncluster) // 4)]
        return iter(ret)

        # num_batch = len(ret) // self.batch_size
        # new_ret = np.array(ret[:(num_batch * self.batch_size)]).reshape(self.batch_size, num_batch)
        # np.random.shuffle(new_ret)
        # return iter(new_ret.reshape(-1).tolist())

    def __len__(self):
        return self.num_identities * self.num_instances


class RandomBatchSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        ret = []
        avai_pids = copy.deepcopy(self.pids)
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            uncluster = 0
            avai_num = self.batch_size
            for pid in selected_pids:
                t = self.index_dic[pid]
                if len(t) == 1:
                    uncluster += 1
                    avai_num -= 1
                avai_pids.remove(pid)
            if uncluster == self.num_pids_per_batch:
                for pid in selected_pids:
                    ret.extend(self.index_dic[pid])
                continue
            # dividers = sorted(random.sample(range(1, avai_num), self.num_pids_per_batch - uncluster - 1))
            # sample_num = [a - b for a, b in zip(dividers + [avai_num], [0] + dividers)]
            m = avai_num
            n = self.num_pids_per_batch - uncluster
            quotient = int(m / n)
            remainder = m % n
            if remainder > 0:
                sample_num = [quotient] * (n - remainder) + [quotient + 1] * remainder
            else:
                sample_num = [quotient] * n
            for pid in selected_pids:
                t = self.index_dic[pid]
                if len(t) == 1:
                    ret.extend(t)
                else:
                    sample = sample_num.pop(0)
                    t = np.random.choice(t, size=sample, replace=(len(t) < sample))
                    ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances