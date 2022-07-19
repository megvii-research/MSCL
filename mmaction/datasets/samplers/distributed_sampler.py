# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmcv.utils import Registry
SAMPLER = Registry('sampler')

class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with a class specific probability, which should be an
    attribute of the dataset (dataset.class_prob, which is a dictionary that
    map label index to the prob). This sampler is only applicable to single
    class recognition dataset. This sampler is also compatible with
    RepeatDataset.

    The default value of dynamic_length is True, which means we use
    oversampling / subsampling, and the dataset length may changed. If
    dynamic_length is set as False, the dataset length is fixed.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 dynamic_length=True,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

        if type(dataset).__name__ == 'RepeatDataset':
            dataset = dataset.dataset

        assert hasattr(dataset, 'class_prob')

        self.class_prob = dataset.class_prob
        self.dynamic_length = dynamic_length
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_indices = defaultdict(list)

        # To be compatible with RepeatDataset
        times = 1
        dataset = self.dataset
        if type(dataset).__name__ == 'RepeatDataset':
            times = dataset.times
            dataset = dataset.dataset
        for i, item in enumerate(dataset.video_infos):
            class_indices[item['label']].append(i)

        if self.dynamic_length:
            indices = []
            for k, prob in self.class_prob.items():
                prob = prob * times
                for i in range(int(prob // 1)):
                    indices.extend(class_indices[k])
                rem = int((prob % 1) * len(class_indices[k]))
                rem_indices = torch.randperm(
                    len(class_indices[k]), generator=g).tolist()[:rem]
                indices.extend(rem_indices)
            if self.shuffle:
                shuffle = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in shuffle]

            # re-calc num_samples & total_size
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # We want to keep the dataloader length same as original
            video_labels = [x['label'] for x in dataset.video_infos]
            probs = [
                self.class_prob[lb] / len(class_indices[lb])
                for lb in video_labels
            ]

            indices = torch.multinomial(
                torch.Tensor(probs),
                self.total_size,
                replacement=True,
                generator=g)
            indices = indices.data.numpy().tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # retrieve indices for current process
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

@SAMPLER.register_module()
class PKDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with a class specific probability, which should be an
    attribute of the dataset (dataset.class_prob, which is a dictionary that
    map label index to the prob). This sampler is only applicable to single
    class recognition dataset. This sampler is also compatible with
    RepeatDataset.

    The default value of dynamic_length is True, which means we use
    oversampling / subsampling, and the dataset length may changed. If
    dynamic_length is set as False, the dataset length is fixed.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 pk_sample_num=2,
                 dynamic_length=True,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

        if type(dataset).__name__ == 'RepeatDataset':
            dataset = dataset.dataset

        self.pk_sample_num = pk_sample_num
        self.total_size *= pk_sample_num
        self.num_samples *= pk_sample_num
        self.dynamic_length = dynamic_length
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        dataset = self.dataset
        metas = dataset.metas
        new_indices = list()
        
        class_indices = defaultdict(list)
        for i, item in enumerate(dataset.metas):
            class_indices[item['label']].append(i)

        # This can not be controlled by torch.generator
        for ind in indices:
            c = metas[ind]['label']
            idxes = class_indices[c].copy()
            idxes.remove(ind)
            pk_sample_idxes = random.sample(idxes, self.pk_sample_num - 1)
            new_indices.append(ind)
            new_indices.extend(pk_sample_idxes)

        # add extra samples to make it evenly divisible
        new_indices += new_indices[:(self.total_size - len(new_indices))]
        assert len(new_indices) == self.total_size, f"{len(new_indices)} vs {self.total_size}"

        # retrieve indices for current process
        new_indices = new_indices[self.rank:self.total_size:self.num_replicas]
        assert len(new_indices) == self.num_samples, f"{len(new_indices)} vs {self.num_samples}"
        return iter(new_indices)
