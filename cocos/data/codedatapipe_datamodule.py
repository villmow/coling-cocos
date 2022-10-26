import omegaconf.errors
import pytorch_lightning as pl
import logging
from typing import Any, Optional, Union

from torch.utils.data import DataLoader

from cocos.datapipes import seeded_worker_init_fn
from cocos.datapipes.common import load_base_dataset, chain_datapipes


log = logging.getLogger(__name__)


class CodeDatapipeDataModule(pl.LightningDataModule):
    """
    Takes a dataset of tokenized source code and a datapipe
    and applies the datapipe to code files within
    the dataset.
    """

    def __init__(self, dataset_cfg, datapipe_cfg):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.datapipe_cfg = datapipe_cfg

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign experiments/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = load_base_dataset(self.dataset_cfg.train.directory)
            train_datapipe_cfg = self.datapipe_cfg.copy()

            # hardcode this config attribute
            try:
                train_datapipe_cfg.batch.max_tokens_in_batch = (
                    self.dataset_cfg.train.num_tokens_in_batch
                )
            except omegaconf.errors.ConfigAttributeError:
                log.error(
                    f"Could not set field 'train_datapipe_cfg.batch.max_tokens_in_batch = {self.dataset_cfg.train.num_tokens_in_batch}'. Missing field"
                )

            dataset = chain_datapipes(dataset, self.datapipe_cfg)
            self.train_dataset = dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "validate" or stage == "fit" or stage is None:
            dataset = load_base_dataset(self.dataset_cfg.valid.directory)

            valid_datapipe_cfg = self.datapipe_cfg.copy()
            try:
                valid_datapipe_cfg.batch.max_tokens_in_batch = (
                    self.dataset_cfg.valid.num_tokens_in_batch
                )
            except omegaconf.errors.ConfigAttributeError:
                log.error(
                    f"Could not set field 'valid_datapipe_cfg.batch.max_tokens_in_batch = {self.dataset_cfg.valid.num_tokens_in_batch}'. Missing field"
                )

            dataset = chain_datapipes(dataset, self.datapipe_cfg)
            self.valid_dataset = dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            dataset = load_base_dataset(self.dataset_cfg.test.directory)

            test_datapipe_cfg = self.datapipe_cfg.copy()
            try:
                test_datapipe_cfg.batch.max_tokens_in_batch = (
                    self.dataset_cfg.test.num_tokens_in_batch
                )
            except omegaconf.errors.ConfigAttributeError:
                log.error(
                    f"Could not set field 'test_datapipe_cfg.batch.max_tokens_in_batch = {self.dataset_cfg.test.num_tokens_in_batch}'. Missing field"
                )

            dataset = chain_datapipes(dataset, test_datapipe_cfg)
            self.test_dataset = dataset

        if stage == "predict":
            raise NotImplementedError

    def get_seed_for_epoch(self, epoch: Optional[int]):
        seed = self.dataset_cfg.seed
        if epoch is not None:
            seed += epoch * 10**6
        return seed

    @property
    def do_pin_memory(self):
        """Only pin memory if we actually use cuda"""
        if self.trainer and (
            getattr(self.trainer, "device_ids", False)
            or getattr(self.trainer, "gpus", False)
        ):
            x = True
        else:
            x = False
        log.info(f"Pin_memory: {x}")
        return x

    def train_dataloader(self):
        current_epoch = self.trainer.current_epoch if self.trainer is not None else 0
        seed = self.get_seed_for_epoch(current_epoch)
        p = self.dataset_cfg.train.num_workers

        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=p,
            worker_init_fn=seeded_worker_init_fn(seed),
            prefetch_factor=self.dataset_cfg.prefetch_factor if p > 0 else 2,
            pin_memory=self.do_pin_memory,
            persistent_workers=True if p > 0 else False,
        )

    def val_dataloader(self):
        p = self.dataset_cfg.valid.num_workers
        return DataLoader(
            self.valid_dataset,
            batch_size=None,
            num_workers=p,
            worker_init_fn=seeded_worker_init_fn(self.dataset_cfg.seed),
            prefetch_factor=self.dataset_cfg.prefetch_factor if p > 0 else 2,
            pin_memory=self.do_pin_memory,
            persistent_workers=True if p > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.dataset_cfg.test.num_workers,
            worker_init_fn=seeded_worker_init_fn(self.dataset_cfg.seed),
            prefetch_factor=self.dataset_cfg.prefetch_factor,
            pin_memory=self.do_pin_memory,
        )
