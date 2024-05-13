# https://github.com/Divadi/SOLOFusion/blob/main/mmdet3d/datasets/samplers/infinite_group_each_sample_in_batch_sampler.py
import itertools
import copy

import numpy as np
import torch
import torch.distributed as dist
from mmengine.dist import get_dist_info, sync_random_seed
from mmdet3d.registry import DATA_SAMPLERS
from torch.utils.data.sampler import Sampler


@DATA_SAMPLERS.register_module()
class GroupInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.

    Shuffling is only done for group order, not done within groups.
    """
    def __init__(
        self,
        sampler,
        batch_size=1,
        world_size=None,
        rank=None,
        seed=0,
        skip_prob=0.5,
        sequence_flip_prob=0.1,
        max_frame_sampling=1,
        skip_iter=0,
    ):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.sampler = sampler
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed()

        self.size = len(self.sampler.dataset)

        assert hasattr(self.sampler.dataset, "flag")
        self.flag = self.sampler.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)
        ]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob
        self.max_frame_sampling = max_frame_sampling
        self.skip_iter = skip_iter
        self._iter = 0
        print(f"[SAMPLER] THE MAX FRAME SAMPLING IS: {self.max_frame_sampling}")
        print(f"[SAMPLER] SKIPPING: {self.skip_iter} ITERS")

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                skip = (
                    np.random.uniform() < self.skip_prob and len(self.buffer_per_local_sample[local_sample_idx]) > 1
                )
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    # skip = False
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[new_group_idx]
                    )

                    # sub sample the sequence to simulate different time difference | done with 0.5 prob otherwise stick to original
                    sub_sample_factor = 1
                    if np.random.uniform() < 0.0:
                        sub_sample_factor = np.random.randint(1, self.max_frame_sampling + 1)

                    self.buffer_per_local_sample[local_sample_idx] = self.buffer_per_local_sample[local_sample_idx
                                                                                                 ][::sub_sample_factor]

                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[local_sample_idx] = self.buffer_per_local_sample[local_sample_idx
                                                                                                     ][::-1]
                    if self.sampler.dataset.keep_consistent_seq_aug:
                        self.aug_per_local_sample[local_sample_idx] = self.sampler.dataset.get_augmentation()

                if not self.sampler.dataset.keep_consistent_seq_aug:
                    self.aug_per_local_sample[local_sample_idx] = self.sampler.dataset.get_augmentation()

                if skip:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                skip_loading = False
                if self._iter < self.skip_iter:
                    skip_loading = True
                curr_batch.append(
                    dict(
                        idx=self.buffer_per_local_sample[local_sample_idx].pop(0),
                        aug_config=self.aug_per_local_sample[local_sample_idx],
                        skip_loading=skip_loading,
                    )
                )
            self._iter += 1
            if self._iter % 1000 == 0 and self._iter < self.skip_iter:
                print(f'skipping iter: {self._iter} / {self.skip_iter}')

            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

