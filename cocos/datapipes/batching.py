from dataclasses import dataclass, field
import logging
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


from cocos.tensortree import collate_tokens

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, functional_datapipe


from cocos.source_code import CodeMeta, CodePair, BatchedCodePairs, BatchedCode, Code
from cocos.source_code import (
    CodePairForSuff,
    BatchedCodeWithDistances,
    BatchedCodePairsWithDistances,
)
from cocos import tokenizer


log = logging.getLogger(__name__)


@dataclass
class Batch:
    max_tokens_in_batch: int
    data: List[Any] = field(default_factory=list, init=False)
    _max_length_in_batch: int = field(default=0, init=False)

    @property
    def max_sequence_length_in_batch(self):
        return self._max_length_in_batch

    @property
    def num_tokens_in_batch(self):
        batch_size = len(self.data)
        return self.max_sequence_length_in_batch * batch_size

    def can_add_sample_to_batch(self, size: int) -> bool:
        longest_size = max(size, self.max_sequence_length_in_batch)
        batch_size = len(self.data) + 1
        return longest_size * batch_size <= self.max_tokens_in_batch

    def add_sample_to_batch(self, sample: Any, size: int):
        self.data.append(sample)
        self._max_length_in_batch = max(self.max_sequence_length_in_batch, size)

    def reset(self):
        self.data = []
        self._max_length_in_batch = 0

    def pop_batch(self) -> List[Any]:
        """Returns the batch and resets everything"""
        batch = self.data
        self.reset()
        return batch

    def is_empty(self) -> bool:
        """No samples currently in batch"""
        return not self.data


@dataclass
class PairBatch(Batch):
    max_tokens_in_batch: int
    data: List[Any] = field(default_factory=list, init=False)

    _max_source_length_in_batch: int = field(default=0, init=False)
    _max_target_length_in_batch: int = field(default=0, init=False)

    @property
    def max_sequence_length_in_batch(self):
        return max(self._max_source_length_in_batch, self._max_target_length_in_batch)

    def can_add_sample_to_batch(self, size: tuple[int, int]) -> bool:
        source_size, target_size = size
        longest_source_size = max(source_size, self.max_sequence_length_in_batch)
        longest_target_size = max(target_size, self.max_sequence_length_in_batch)
        longest_size = max(longest_source_size, longest_target_size)

        batch_size = len(self.data) + 1

        return longest_size * batch_size <= self.max_tokens_in_batch

    def add_sample_to_batch(self, sample: Any, size: tuple[int, int]):
        source_size, target_size = size
        self.data.append(sample)
        self._max_source_length_in_batch = max(
            self._max_source_length_in_batch, source_size
        )
        self._max_target_length_in_batch = max(
            self._max_target_length_in_batch, target_size
        )

    def reset(self):
        self._max_source_length_in_batch = 0
        self._max_target_length_in_batch = 0
        super().reset()


@dataclass
class BucketedBatch:
    max_tokens_in_batch: int
    num_buckets: int

    max_sequence_length: int

    # buckets are stored in bucket list and in a mapping from sequence length to bucket (reference)
    _buckets: List[Any] = field(init=False)
    estimated_bucket_size: int = field(init=False)

    @property
    def buckets(self) -> Iterable[Batch]:
        return self._buckets

    def __post_init__(self):
        if self.max_tokens_in_batch < self.max_sequence_length:
            raise ValueError(
                "max_tokens_in_batch should be at least max_sequence length."
            )

        # last bucket may be larger
        self.estimated_bucket_size = self.max_sequence_length // self.num_buckets
        self._buckets = self.init_buckets()

    def init_buckets(self) -> list[Batch]:
        return [Batch(self.max_tokens_in_batch) for _ in range(self.num_buckets)]

    def check_if_valid_size(self, size: int) -> bool:
        if size == 0:
            return False
        elif size > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {size} > {self.max_sequence_length} max sequence length. "
                f"Shorten or filter before (FilterSizeDataset)!"
            )
        return True

    def get_bucket_for_size(self, size: int):
        bucket_idx = size // self.estimated_bucket_size

        if bucket_idx >= self.num_buckets:
            bucket_idx = self.num_buckets - 1  # put in last bucket

        return self._buckets[bucket_idx]

    def add_sample(self, sample: Any, size) -> Optional[List[Any]]:
        """Adds a sample and returns the batch, if it is full"""
        if not self.check_if_valid_size(size):
            return

        if isinstance(size, tuple) and isinstance(size[0], torch.Tensor):
            size = tuple(s.item() for s in size)
        elif isinstance(size, torch.Tensor):
            size = size.item()

        bucket = self.get_bucket_for_size(size)

        if bucket.can_add_sample_to_batch(size):
            bucket.add_sample_to_batch(sample, size)
        else:
            # batch full,
            batch = bucket.pop_batch()
            bucket.add_sample_to_batch(sample, size)

            assert batch, "batch should not be empty"

            return batch

    def get_batches(self):
        """Returns all remaining batches"""
        for bucket in self.buckets:
            batch = bucket.pop_batch()

            if batch:
                yield batch


@dataclass
class PairBucketedBatch(BucketedBatch):
    # buckets are stored in bucket list and in a mapping from sequence length to bucket (reference)
    # only type is changing
    _buckets: List[List[PairBatch]] = field(init=False)

    @property
    def buckets(self) -> Iterable[PairBatch]:
        return (batch for bucket in self._buckets for batch in bucket)

    def check_if_valid_size(self, size: tuple[int, int]) -> bool:
        source_size, target_size = size
        if source_size == 0 or target_size == 0:
            return False
        elif source_size > self.max_sequence_length:
            raise ValueError(
                f"Source sequence length {source_size} > {self.max_sequence_length} max sequence length. "
                f"Shorten or filter before (FilterSizeDataset)!"
            )
        elif target_size > self.max_sequence_length:
            raise ValueError(
                f"Source sequence length {target_size} > {self.max_sequence_length} max sequence length. "
                f"Shorten or filter before (FilterSizeDataset)!"
            )
        return True

    def get_bucket_for_size(self, size: tuple[int, int]):
        source_size, target_size = size
        source_bucket = source_size // self.estimated_bucket_size
        target_bucket = target_size // self.estimated_bucket_size

        if source_bucket >= self.num_buckets:
            source_bucket = self.num_buckets - 1  # put in last bucket
        if target_bucket >= self.num_buckets:
            target_bucket = self.num_buckets - 1  # put in last bucket

        return self._buckets[source_bucket][target_bucket]

    def init_buckets(self):
        return [
            [PairBatch(self.max_tokens_in_batch) for _ in range(self.num_buckets)]
            for _ in range(self.num_buckets)
        ]


@functional_datapipe("batch_pairs")
class CodePairBatcherIterDataPipe(IterDataPipe):
    """
    Creates batches with minimum amount of padding, by placing samples in Buckets grouped by their size.
    """

    def __init__(
        self,
        datapipe: Iterable[CodePair],
        collate_fn: Optional[Callable],
        *,
        max_tokens_in_batch: int,
        max_sequence_length: int,
        num_buckets: int,
    ):
        super().__init__()

        self.datapipe = datapipe
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        self.buckets = PairBucketedBatch(
            max_tokens_in_batch=max_tokens_in_batch,
            max_sequence_length=max_sequence_length,
            num_buckets=num_buckets,
        )

    def __iter__(self):
        try:
            for sample in self.datapipe:
                batch = self.buckets.add_sample(
                    sample, sample.size
                )  # returns a full batch if ready
                if batch is not None:
                    yield self.collate_fn(batch)

            # yield unfinalized batches
            for batch in self.buckets.get_batches():
                yield self.collate_fn(batch)
        except Exception as e:
            log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
            import traceback

            log.error(traceback.format_exc())


class CodePairsCollater:
    def __init__(
        self,
        left_pad_source: bool,
        left_pad_target: bool,
        input_feeding: bool = True,
        pad_id: int = tokenizer.get_pad_id(),
        eos_id: int = tokenizer.get_eos_id(),
    ):
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.input_feeding = input_feeding
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __call__(self, samples: List[Union[CodePair, CodePairForSuff]]):
        try:
            if isinstance(samples[0], CodePairForSuff):
                return collate_suff_code_pairs(
                    samples,
                    self.left_pad_source,
                    self.left_pad_target,
                    self.input_feeding,
                    self.pad_id,
                    self.eos_id,
                )
            else:
                return collate_code_pairs(
                    samples,
                    self.left_pad_source,
                    self.left_pad_target,
                    self.input_feeding,
                    self.pad_id,
                    self.eos_id,
                )
        except Exception as e:
            log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
            import traceback

            log.error(traceback.format_exc())


def collate_code_pairs(
    samples: List[CodePair],
    left_pad_source: bool,
    left_pad_target: bool,
    input_feeding: bool = True,
    pad_id: int = tokenizer.get_pad_id(),
    eos_id: int = tokenizer.get_eos_id(),
) -> Optional[BatchedCodePairs]:
    if len(samples) == 0:
        return

    src_tokens = collate_tokens(
        [s.source.data for s in samples],
        pad_idx=pad_id,
        left_pad=left_pad_source,
        pad_to_length=None,
    )

    # sort by descending source length
    src_lengths = (
        torch.tensor([s.source.data.size(0) for s in samples]).long().squeeze()
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)

    # collate target tokens
    tgt_tokens = collate_tokens(
        [s.target.data for s in samples],
        pad_idx=pad_id,
        left_pad=left_pad_target,
        pad_to_length=None,
    )
    tgt_tokens = tgt_tokens.index_select(0, sort_order)
    tgt_lengths = (
        torch.tensor([s.target.data.size(0) for s in samples])
        .squeeze()
        .long()
        .index_select(0, sort_order)
    )
    ntokens = tgt_lengths.sum().item()

    prev_output_tokens = None
    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        assert all(
            s.target.data[-1] == eos_id for s in samples
        ), "all targets should end with EOS"
        prev_output_tokens = collate_tokens(
            [torch.roll(s.target.data, shifts=1) for s in samples],
            pad_idx=pad_id,
            left_pad=left_pad_target,
            pad_to_length=None,
        )

    source = BatchedCode(
        tokens=src_tokens,
        lengths=src_lengths,
        meta=[s.source.meta for s in samples],
        attention_mask=(src_tokens != pad_id),
        nsamples=len(src_tokens),
        ntokens=src_lengths.sum().item(),
    )
    target = BatchedCode(
        tokens=tgt_tokens,
        lengths=tgt_lengths,
        meta=[s.target.meta for s in samples],
        attention_mask=(tgt_tokens != pad_id),
        nsamples=len(tgt_tokens),
        ntokens=tgt_lengths.sum().item(),
        shifted_tokens=prev_output_tokens,
    )

    batch = BatchedCodePairs(source=source, target=target)

    return batch


def collate_tokens_2d(values: list[torch.Tensor], pad_id, left_pad: bool = False):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size1 = max(v.size(0) for v in values)
    size2 = max(v.size(1) for v in values)

    res = values[0].new(len(values), size1, size2).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size1 - v.size(0) :, size2 - v.size(1) :]
            if left_pad
            else res[i][: v.size(0), : v.size(1)],
        )

    return res


def collate_suff_code_pairs(
    samples: List[CodePairForSuff],
    left_pad_source: bool,
    left_pad_target: bool,
    input_feeding: bool = True,
    pad_id: int = tokenizer.get_pad_id(),
    eos_id: int = tokenizer.get_eos_id(),
) -> Optional[BatchedCodePairsWithDistances]:
    if len(samples) == 0:
        return

    src_tokens = collate_tokens(
        [s.source.data for s in samples],
        pad_idx=pad_id,
        left_pad=left_pad_source,
        pad_to_length=None,
    )

    # sort by descending source length
    src_lengths = (
        torch.tensor([s.source.data.size(0) for s in samples]).long().squeeze()
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_attention_mask = collate_tokens_2d(
        [s.source.attention_mask for s in samples], pad_id=False, left_pad=False
    )
    src_distances = collate_tokens_2d(
        [s.source.distances for s in samples], pad_id=0, left_pad=False
    )

    # collate target tokens
    tgt_tokens = collate_tokens(
        [s.target.data for s in samples],
        pad_idx=pad_id,
        left_pad=left_pad_target,
        pad_to_length=None,
    )
    tgt_tokens = tgt_tokens.index_select(0, sort_order)
    tgt_lengths = (
        torch.tensor([s.target.data.size(0) for s in samples])
        .squeeze()
        .long()
        .index_select(0, sort_order)
    )
    ntokens = tgt_lengths.sum().item()
    tgt_attention_mask = collate_tokens_2d(
        [s.target.attention_mask for s in samples], pad_id=False, left_pad=False
    )
    tgt_distances = collate_tokens_2d(
        [s.target.distances for s in samples], pad_id=0, left_pad=False
    )

    prev_output_tokens = None
    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        assert all(
            s.target.data[-1] == eos_id for s in samples
        ), "all targets should end with EOS"
        prev_output_tokens = collate_tokens(
            [torch.roll(s.target.data, shifts=1) for s in samples],
            pad_idx=pad_id,
            left_pad=left_pad_target,
            pad_to_length=None,
        )

    source = BatchedCodeWithDistances(
        tokens=src_tokens,
        lengths=src_lengths,
        meta=[s.source.meta for s in samples],
        attention_mask=src_attention_mask,  # !
        nsamples=len(src_tokens),
        ntokens=src_lengths.sum().item(),
        distances=src_distances,
    )
    target = BatchedCodeWithDistances(
        tokens=tgt_tokens,
        lengths=tgt_lengths,
        meta=[s.target.meta for s in samples],
        attention_mask=tgt_attention_mask,
        nsamples=len(tgt_tokens),
        ntokens=tgt_lengths.sum().item(),
        shifted_tokens=prev_output_tokens,
        distances=tgt_distances,
    )

    batch = BatchedCodePairsWithDistances(source=source, target=target)
    return batch


def batch_code_pairs(
    dataset: Iterable,
    max_tokens_in_batch: int,
    num_buckets: int,
    max_sequence_length: int,
):
    from functools import partial

    collate = partial(
        collate_code_pairs,
        input_feeding=True,
        left_pad_source=False,
        left_pad_target=False,
    )

    return dataset.batch_pairs(
        max_tokens_in_batch=max_tokens_in_batch,
        max_sequence_length=max_sequence_length,
        num_buckets=num_buckets,
        collate_fn=collate,
        size_fn=lambda s: len(s.source.data),
    )


class CodeBatcherIterDataPipe(IterDataPipe):
    """
    Creates batches with minimum amount of padding, by placing samples in Buckets grouped by their size.
    """

    def __init__(
        self,
        datapipe: Iterable[Code],
        collate_fn: Optional[Callable],
        *,
        max_tokens_in_batch: int,
        max_sequence_length: int,
        num_buckets: int,
    ):
        super().__init__()

        self.datapipe = datapipe
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        self.buckets = BucketedBatch(
            max_tokens_in_batch=max_tokens_in_batch,
            max_sequence_length=max_sequence_length,
            num_buckets=num_buckets,
        )

    def __iter__(self):
        try:
            for sample in self.datapipe:
                batch = self.buckets.add_sample(
                    sample, sample.size
                )  # returns a full batch if ready
                if batch is not None:
                    yield self.collate_fn(batch)

            # yield unfinalized batches
            for batch in self.buckets.get_batches():
                yield self.collate_fn(batch)
        except Exception as e:
            log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
            import traceback

            log.error(traceback.format_exc())


class CodeCollater:
    def __init__(
        self,
        left_pad: bool,
        pad_id: int = tokenizer.get_pad_id(),
    ):
        self.left_pad = left_pad
        self.pad_id = pad_id

    def __call__(self, samples: List[CodePair]):
        try:
            return collate_code(samples, self.left_pad, self.pad_id)
        except Exception as e:
            log.error(f"[{self.__class__.__name__}] Caught exception: {repr(e)}")
            import traceback

            log.error(traceback.format_exc())


def collate_code(
    samples: List[Code],
    left_pad: bool,
    pad_id: int = tokenizer.get_pad_id(),
) -> Optional[BatchedCode]:
    if len(samples) == 0:
        return

    src_tokens = collate_tokens(
        [s.data for s in samples],
        pad_idx=pad_id,
        left_pad=left_pad,
        pad_to_length=None,
    )

    # sort by descending source length
    src_lengths = torch.tensor([s.data.size(0) for s in samples]).long().squeeze()
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)

    return BatchedCode(
        tokens=src_tokens,
        lengths=src_lengths,
        meta=[samples[sample_idx].meta for sample_idx in sort_order],
        attention_mask=(src_tokens != pad_id),
        nsamples=len(src_tokens),
        ntokens=src_lengths.sum().item(),
    )
