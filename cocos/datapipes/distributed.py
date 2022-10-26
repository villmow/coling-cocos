from itertools import count
import logging
import random
from typing import (
    Any,
    Iterable,
    List,
    Dict,
    Optional,
    Callable,
    Sequence,
    Generator,
    Union,
    Iterator,
)

import numpy as np

import torch
from torch.utils.data import IterDataPipe, Dataset, IterableDataset
from cocos.datapipes.common import SeedMixin


log = logging.getLogger(__name__)


class ShardedDataset(IterDataPipe):
    def __init__(self, dataset: Dataset, infinite: bool = False):
        """
        Infinite will not stop yielding from the base dataset. This is useful, when each worker/gpu
        produces different amount of batches per epoch (making DDP fail). Note, that datasets are
        not informed about an epoch change (proper seeding may not work!)!

        :param dataset:
        :param infinite:
        """
        super().__init__()

        self.dataset = dataset
        self.infinite = infinite

        assert not isinstance(
            self.dataset, IterableDataset
        ), "Sharding does not support iterable dataset"

        self.worker_id = None
        self.rank = None
        self.num_workers = None
        self.world_size = None

    @property
    def total_num_workers(self):
        if self.num_workers is None or self.worker_id is None or self.rank is None:
            log.warning(
                "[ShardedDataset] num_workers, worker_id, rank are not set and "
                "each worker will yield the full dataset. "
                "Pass worker_init_function to dataloader!"
            )
            self.worker_id = 0
            self.num_workers = 1
            self.world_size = 1
            self.rank = 0

        return self.num_workers * self.world_size

    @property
    def shard_id(self):
        if self.num_workers is None or self.worker_id is None or self.rank is None:
            log.warning(
                "[ShardedDataset] num_workers, worker_id, rank are not set and "
                "each worker will yield the full dataset. "
                "Pass worker_init_function to dataloader!"
            )
            self.worker_id = 0
            self.num_workers = 1
            self.world_size = 1

        rank = self.rank if self.world_size > 1 else 0

        shard_id = self.worker_id + (rank * self.num_workers)
        return shard_id

    def sharded_data(self) -> Generator[Any, None, None]:
        log.info(
            f"[Shard {self.shard_id}] Start yielding data in "
            f"worker: {self.worker_id}, "
            f"total_num_workers: {self.total_num_workers}, "
            f"num_workers: {self.num_workers}, "
            f"rank: {self.rank}, "
            f"world_size: {self.world_size}"
        )

        for i in count(self.shard_id, self.total_num_workers):
            try:
                yield self.dataset[i]
            except IndexError:
                return

    def __iter__(self):
        if self.infinite:
            for iteration in count():
                yield from self.sharded_data()
                log.info(
                    f"[Shard {self.shard_id}, R{self.rank}, W{self.worker_id}] Finished iteration {iteration}."
                )
        else:
            yield from self.sharded_data()
            log.warning(
                f"[Shard {self.shard_id}, R{self.rank}, W{self.worker_id}] [IndexError] Reached end of dataset. Stopping."
            )


def _find_datasets(dataset) -> Generator[Dataset, None, None]:
    # check if we have a dataset in our datapipe
    if isinstance(dataset, Dataset):
        yield dataset

    def check_dict(some_dict: dict):
        for attr, value in some_dict.items():
            if isinstance(value, Dataset):
                yield from _find_datasets(value)
            elif isinstance(value, (list, tuple, set)):
                for subitem in value:
                    if isinstance(subitem, Dataset):
                        yield from _find_datasets(subitem)
            elif isinstance(value, dict):
                yield from check_dict(value)

    yield from check_dict(dataset.__dict__)


def worker_init_fn(worker_id, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        raise ValueError(
            "Set seed argument explicitly with functools partial on this worker_init_fn."
        )

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    # check if we are in a distributed setting
    try:
        import torch.distributed as dist

        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except RuntimeError as e:
        # no distributed training:
        rank, world_size = 1, 1

    # configure the dataset to only process the split workload
    worker_id2 = worker_info.id
    assert worker_id == worker_id2

    log.info(
        f"[R{rank},W{worker_id}] Initializing worker {worker_id} for rank {rank} ({world_size} gpus)"
    )

    num_workers = worker_info.num_workers

    # find sharded dataset and set attributes

    for ds in _find_datasets(dataset):
        if isinstance(ds, ShardedDataset):
            sharded_dataset = ds
            sharded_dataset.worker_id = worker_id
            sharded_dataset.num_workers = num_workers
            sharded_dataset.world_size = world_size
            sharded_dataset.rank = rank
            log.info(
                f"[Shard {sharded_dataset.shard_id}] Initialized worker "
                f"worker: {sharded_dataset.worker_id}, num_workers: {sharded_dataset.num_workers}, "
                f"rank: {sharded_dataset.rank}, world_size: {sharded_dataset.world_size}"
            )
        elif isinstance(ds, SeedMixin):
            ds.seed = seed


def seeded_worker_init_fn(seed):
    from functools import partial

    return partial(worker_init_fn, seed=seed)
