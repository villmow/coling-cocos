from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Union, Sequence, Any, Tuple

import numpy as np
from cocos import tensortree
import torch

from cocos import to_tree
from cocos import tokenizer


@dataclass
class CodeMeta:
    language: str
    repository: Optional[str]
    path: Optional[str]

    @classmethod
    def from_dict(cls, sample: dict):
        return cls(sample["language"], sample["repository"], sample["path"])

    def __str__(self):
        return (
            f"Repository: {self.repository}\n"
            f"Filepath:   {self.file_url}\n"
            f"Language:   {self.language}"
        )

    @property
    def repo_name(self):
        return self.repository.split("/")[1]

    @property
    def repo_url(self):
        return f"https://www.github.com/{self.repository}"

    @property
    def branch(self):
        directory_name = self.path.split("/")[0]
        branch = directory_name.replace(f"{self.repo_name}-", "")
        return branch

    @property
    def path_from_root(self):
        return self.path.split("/", maxsplit=1)[1]

    @property
    def file_url(self):
        return f"https://www.github.com/{self.repository}/blob/{self.branch}/{self.path_from_root}"


@dataclass
class Code:
    data: Union[torch.Tensor, tensortree.TensorTree]
    meta: CodeMeta

    def __str__(self):
        t = tokenizer.load_tokenizer()
        code = t.decode(self.data)
        return f"{self.meta}\n" f"Code ({len(self.data)} tokens): \n" f"{code}"

    def __len__(self):
        return len(self.data)

    @property
    def size(self) -> int:
        if getattr(self, "_size", None) is None:
            self._size = len(self)

        return self._size


@dataclass
class CodeTree(Code):
    data: tensortree.TensorTree

    @property
    def size(self) -> int:
        if getattr(self, "_size", None) is None:
            self._size = self.data.leaves_mask().sum().item()

        return self._size

    @property
    def tree(self) -> tensortree.TensorTree:
        """alias for data"""
        return self.data

    @classmethod
    def from_dict(cls, sample: dict):
        if "tree" in sample:
            tree = sample["tree"]
        else:
            tree = to_tree(sample)

        meta = CodeMeta.from_dict(sample)
        return cls(tree, meta)

    def str_tree(self, keep_bpe: bool = True):
        t = tokenizer.load_tokenizer()
        tree = t.decode_tree(self.data, keep_bpe=keep_bpe)
        return f"{self.meta}\n" f"Code: \n" f"{tree.pformat()}"


@dataclass
class TruncatedCodeTree(CodeTree):
    """A larger source_code file, which has been shortened."""

    cutout_indices: torch.Tensor  # indices at which the tree has been truncated
    old_cutout_indices: Optional[
        torch.Tensor
    ] = None  # indices in which the original tree has been truncated


@dataclass
class ExtractedCodeTree(CodeTree):
    """Some piece of source_code, extracted from a file."""

    cutout_index: Union[torch.Tensor, int]
    replacement: Union[torch.Tensor, int]


@dataclass
class CodePair:
    """Paired source_code tokens."""

    source: Code
    target: Code

    def __str__(self):
        return (
            f"{'#' * 40} Source {'#' * 40}\n"
            f"{self.source}\n"
            f"{'#' * 40} Target {'#' * 40}\n"
            f"{self.target}\n"
            f"{'#' * 88}"
        )

    @property
    def size(self) -> tuple[int, int]:
        return (
            self.source.size,
            self.target.size,
        )


@dataclass
class CodeTreePair(CodePair):
    """Paired source_code tokens."""

    source: CodeTree
    target: CodeTree


@dataclass
class CodeWithSearchResults:
    code: Code
    results: list[Code]
    scores: Union[list[float], np.ndarray]

    def __str__(self):
        s = "#" * 100
        s += f"{self.code}\n"
        s += "#" * 100
        for i, (result, score) in enumerate(zip(self.results, self.scores)):
            s += f"{i}. Score: {score}\n"
            s += f"{result}\n"
            s += "-" * 100
        s += "#" * 100
        return s

    @property
    def size(self) -> int:
        return self.code.size


@dataclass
class CodePairWithSearchResults(CodePair):
    source: CodeWithSearchResults
    target: Code


@dataclass
class CodeForSuFF(Code):
    distances: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class CodePairForSuff(CodePair):
    source: CodeForSuFF
    target: CodeForSuFF


@dataclass
class BatchedCode:
    tokens: torch.Tensor
    lengths: torch.Tensor
    attention_mask: torch.Tensor
    meta: List[CodeMeta]

    ntokens: int
    nsamples: int

    shifted_tokens: Optional[torch.Tensor] = None

    @property
    def device(self):
        return self.tokens.device

    def pin_memory(self):
        self.tokens = self.tokens.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()

        if self.shifted_tokens is not None:
            self.shifted_tokens = self.shifted_tokens.pin_memory()

        return self

    def to(self, device: torch.device, **kwargs):
        if self.tokens.device == device:
            return self

        return BatchedCode(
            tokens=self.tokens.to(device, **kwargs),
            lengths=self.lengths.to(device, **kwargs),
            attention_mask=self.attention_mask.to(device, **kwargs),
            meta=self.meta,
            ntokens=self.ntokens,
            nsamples=self.ntokens,
            shifted_tokens=(
                self.shifted_tokens.to(device, **kwargs)
                if self.shifted_tokens is not None
                else self.shifted_tokens
            ),
        )


@dataclass
class BatchedCodePairs:
    source: BatchedCode
    target: BatchedCode

    def pin_memory(self):
        self.source = self.source.pin_memory()
        self.target = self.target.pin_memory()

        return self

    def to(self, device: torch.device, **kwargs):
        if self.source.device == device:
            return self

        return BatchedCodePairs(
            source=self.source.to(device, **kwargs),
            target=self.target.to(device, **kwargs),
        )


@dataclass
class BatchedCodeWithDistances(BatchedCode):
    distances: torch.Tensor = None  # should not be None!

    def pin_memory(self):
        self.distances = self.distances.pin_memory()

        self.tokens = self.tokens.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()

        if self.shifted_tokens is not None:
            self.shifted_tokens = self.shifted_tokens.pin_memory()

        return self

    def to(self, device: torch.device, **kwargs):
        if self.tokens.device == device:
            return self

        return BatchedCodeWithDistances(
            tokens=self.tokens.to(device, **kwargs),
            lengths=self.lengths.to(device, **kwargs),
            attention_mask=self.attention_mask.to(device, **kwargs),
            meta=self.meta,
            ntokens=self.ntokens,
            nsamples=self.ntokens,
            shifted_tokens=(
                self.shifted_tokens.to(device, **kwargs)
                if self.shifted_tokens is not None
                else self.shifted_tokens
            ),
            distances=self.distances.to(device, **kwargs),
        )


@dataclass
class BatchedCodePairsWithDistances(BatchedCodePairs):
    source: BatchedCodeWithDistances
    target: BatchedCodeWithDistances

    def to(self, device: torch.device, **kwargs):
        if self.source.device == device:
            return self

        return BatchedCodePairsWithDistances(
            source=self.source.to(device, **kwargs),
            target=self.target.to(device, **kwargs),
        )


@dataclass
class FiDBatch:
    source: torch.Tensor  # [B, P, PL]  P = query with passage concatenated
    source_attention_mask: torch.Tensor

    target: torch.Tensor  # [B, TL]
    target_attention_mask: torch.Tensor  # [B, TL]

    source_length: list[int]
    passage_lengths: list[list[int]]

    source_meta: list[CodeMeta]
    target_meta: list[CodeMeta]
    passages_meta: list[list[CodeMeta]]

    @property
    def device(self):
        return self.tokens.device

    def pin_memory(self):
        self.source = self.source.pin_memory()
        self.source_attention_mask = self.source_attention_mask.pin_memory()

        self.target = self.target.pin_memory()
        self.target_attention_mask = self.target_attention_mask.pin_memory()

        return self

    def to(self, device: torch.device, **kwargs):
        if self.source.device == device:
            return self

        return FiDBatch(
            source=self.source.to(device, **kwargs),
            source_attention_mask=self.source_attention_mask.to(device, **kwargs),
            target=self.target.to(device, **kwargs),
            target_attention_mask=self.target_attention_mask.to(device, **kwargs),
            source_length=self.source_length,
            source_meta=self.source_meta,
            target_meta=self.target_meta,
            passage_lengths=self.passage_lengths,
            passages_meta=self.passages_meta,
        )

    def format_full(self) -> str:
        t = tokenizer.load_tokenizer()
        bsz, num_queries, length = self.source.shape
        log_lines = []

        for sample_idx in range(bsz):
            log_lines.append("#" * 100)

            queries = []
            for i, query_passage in enumerate(self.source[sample_idx]):
                query = t.decode(
                    query_passage[self.source_attention_mask[sample_idx, i]]
                )
                queries.append(query)
            target = t.decode(
                self.target[sample_idx][self.target_attention_mask[sample_idx]]
            )

            log_lines.append(f" Query meta: {self.source_meta[sample_idx]}")
            log_lines.append("-" * 100)
            log_lines.append(f"TARGET: \n{target}")
            if self.log_tensors:
                log_lines.append(
                    f"TARGET: \n{self.target[sample_idx][self.target_attention_mask[sample_idx]]}"
                )
            log_lines.append("-" * 100)

            for idx, (s, meta) in enumerate(
                zip(queries, self.passages_meta[sample_idx])
            ):
                log_lines.append("-" * 100)
                log_lines.append(f"PASSAGE {idx} {meta}")
                log_lines.append(s)
            log_lines.append("#" * 100)
        return "\n".join(log_lines)

    def format_pretty(self) -> str:
        t = tokenizer.load_tokenizer()
        bsz, num_queries, length = self.source.shape
        log_lines = []

        for sample_idx in range(bsz):
            log_lines.append("#" * 100)

            context = None
            passages = []
            for i, query_passage in enumerate(self.source[sample_idx]):
                query = t.decode(
                    query_passage[self.source_attention_mask[sample_idx, i]]
                )

                context, passage = query.split("[SEP]", 1)
                passages.append(passage)

            target = t.decode(
                self.target[sample_idx][self.target_attention_mask[sample_idx]]
            )

            log_lines.append("#" * 100)
            log_lines.append(str(self.source_meta[sample_idx]))
            log_lines.append(f"CONTEXT: \n{context}")
            log_lines.append("-" * 100)
            log_lines.append(f"TARGET: \n{target}")
            log_lines.append("-" * 100)

            for idx, (s, meta) in enumerate(
                zip(passages, self.passages_meta[sample_idx])
            ):
                log_lines.append(f"PASSAGE {idx} \n{meta}")
                log_lines.append("-" * 100)
                log_lines.append(s)
                log_lines.append("-" * 100)
            log_lines.append("#" * 100)
        return "\n".join(log_lines)

    def __str__(self):
        return self.format_pretty()
