import torch
import random
from typing import Optional, Iterator, List

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

import math


class RandomSupervisionSampler(Sampler):
    def __init__(self, dataset: Dataset, data_type_sampling_probability: List[float], 
                 batch_size: int = 32, seed: int = 42) -> None:
        self.dataset = dataset
        self.data_type_sampling_probability = torch.tensor(data_type_sampling_probability).float()
        self.batch_size = batch_size
        self.seed = seed

        self.data_source_len = len(dataset)
        self.num_batch_iters = math.ceil(self.data_source_len / self.batch_size)
        self.total_size = self.batch_size * self.num_batch_iters
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        coin_tosses = torch.rand(size=(self.num_batch_iters,), generator=g)
        indices = []

        for coin_toss in coin_tosses:
            if coin_toss <= self.data_type_sampling_probability[0]:
                # Sample from fully supervised data (x, z)
                indices.extend(self._sample_indices(0, self.dataset.sup_len_xz, self.batch_size, g))
            elif coin_toss <= self.data_type_sampling_probability[0] + self.data_type_sampling_probability[1]:
                # Sample from partially supervised data (x only)
                indices.extend(self._sample_indices(self.dataset.sup_len_xz, 
                                                    self.dataset.sup_len_xz + self.dataset.sup_len_x, 
                                                    self.batch_size, g))
            else:
                # Sample from unsupervised data
                indices.extend(self._sample_indices(self.dataset.sup_len_xz + self.dataset.sup_len_x, 
                                                    self.data_source_len, 
                                                    self.batch_size, g))

        return iter(indices[:self.total_size])

    def _sample_indices(self, start: int, end: int, size: int, generator: torch.Generator) -> List[int]:
        if start >= end:
            raise ValueError(f"Invalid range: start ({start}) must be less than end ({end})")
        return torch.randint(start, end, size=(size,), generator=generator).tolist()

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

class RandomSupervisionSamplerDDP(DistributedSampler):
    def __init__(self, dataset: Dataset, data_type_sampling_probability: List[float],
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False,
                 batch_size: int = 32) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        self.data_type_sampling_probability = torch.tensor(data_type_sampling_probability).float()
        self.batch_size = batch_size
        
        self.replica_size = self.batch_size * self.num_replicas
        self.num_batch_iter = math.ceil(len(dataset) / self.replica_size)
        self.total_size = self.num_batch_iter * self.replica_size

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        coin_tosses = torch.rand(size=(self.num_batch_iter,), generator=g)
        indices = []

        for coin_toss in coin_tosses:
            if coin_toss <= self.data_type_sampling_probability[0]:
                # Sample from fully supervised data (x, z)
                indices.extend(self._sample_indices(0, self.dataset.sup_len_xz, self.replica_size, g))
            elif coin_toss <= self.data_type_sampling_probability[0] + self.data_type_sampling_probability[1]:
                # Sample from partially supervised data (x only)
                indices.extend(self._sample_indices(self.dataset.sup_len_xz, 
                                                    self.dataset.sup_len_xz + self.dataset.sup_len_x, 
                                                    self.replica_size, g))
            else:
                # Sample from unsupervised data
                indices.extend(self._sample_indices(self.dataset.sup_len_xz + self.dataset.sup_len_x, 
                                                    len(self.dataset), 
                                                    self.replica_size, g))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

    def _sample_indices(self, start: int, end: int, size: int, generator: torch.Generator) -> List[int]:
        if start >= end:
            raise ValueError(f"Invalid range: start ({start}) must be less than end ({end})")
        return torch.randint(start, end, size=(size,), generator=generator).tolist()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)