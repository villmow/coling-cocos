from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)


class SeedMixin:
    @property
    def seed(self):
        return getattr(self, "_seed", None)

    @seed.setter
    def seed(self, seed: int):
        log.debug(
            f"[{self.__class__.__name__}] Changing seed from {self.seed} to {seed}"
        )
        self._seed = seed
        self.on_seed_change(seed)

    def on_seed_change(self, seed):
        pass


def load_base_dataset(directory):
    """Directory can either be Huggingface Dataset or a Directory with many dataset objects as subdirectories"""
    import datasets
    from torch.utils.data import ConcatDataset

    directory = Path(directory)
    try:
        dataset = datasets.load_from_disk(str(directory))
    except FileNotFoundError:
        dataset = ConcatDataset(
            (datasets.load_from_disk(str(d)) for d in directory.iterdir() if d.is_dir())
        )
    return dataset


def chain_datapipes(dataset, datapipe_cfg: DictConfig):
    for datapipe_name in datapipe_cfg:
        print(f"Instantiating datapipe: {datapipe_name}")
        dataset = hydra.utils.instantiate(datapipe_cfg[datapipe_name], dataset)

    return dataset


from cocos.source_code import CodePair, Code, CodeTree
import torch
from typing import List, Optional, Union


class CodePairFilter:
    def __init__(self, max_source_size: int, max_target_size: int):
        self.max_source_size = max_source_size
        self.max_target_size = max_target_size

    def __call__(self, sample: CodePair):
        return filter_code_pair(sample, self.max_source_size, self.max_target_size)


class TargetLargerThanSourceFilter:
    def __call__(self, sample: CodePair):
        if sample.source.size < sample.target.size:
            log.warning(str(sample))
            log.warning(
                f"[{self.__class__.__name__}] Filtered sample with larger target than source:"
            )
            return False

        return True


class LanguageFilter:
    def __init__(self, supported_languages: List[str]):
        self.languages = supported_languages

    def __call__(self, sample: CodePair):
        return sample.source.meta.language in self.languages


def filter_code_pair(sample: CodePair, max_source_size: int, max_target_size: int):
    if sample.source.size > max_source_size:
        return False
    if sample.target.size > max_target_size:
        return False

    return True


class NumberOfLinesFilter:
    def __init__(
        self, min_numer_of_lines: int = 0, max_numer_of_lines: Optional[int] = None
    ):
        self.min_numer_of_lines = min_numer_of_lines
        self.max_numer_of_lines = max_numer_of_lines

    def __call__(self, sample: Union[torch.Tensor, Code, CodeTree]):
        if isinstance(sample, torch.Tensor):
            token_ids = sample
        elif isinstance(sample, CodeTree):
            token_ids = sample.tree.node_data
        elif isinstance(sample, Code):
            token_ids = sample.data
        else:
            log.warning(
                f"Sample filtered by number of lines filter. Class {sample.__class__.__name__} not implemented."
            )
            return False

        newline_mask = token_ids == 10
        double_newline_mask = token_ids == 9

        number_of_newlines = newline_mask.sum() + (double_newline_mask.sum() * 2)
        if self.max_numer_of_lines is not None:
            return (
                self.min_numer_of_lines <= number_of_newlines <= self.max_numer_of_lines
            ).item()
        else:
            return (self.min_numer_of_lines <= number_of_newlines).item()
