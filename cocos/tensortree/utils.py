from itertools import takewhile, repeat
import pathlib as pl
import os
from typing import Union, List, Optional, Sequence, Any

import numpy as np
import torch

import functools
from cocos import tensortree


DONT_VALIDATE_INDEX = os.getenv("DONT_VALIDATE_INDEX", False)
DEBUG = False


def validate_index(_func=None, allow_none: bool = False):
    """Should be only used inside TensorTree and on functions that receive
    a node_idx as the first argument after self.
    """

    def decorator(func):
        if DONT_VALIDATE_INDEX:
            return func

        @functools.wraps(func)
        def wrapper(self, node_idx, *args, **kwargs):
            if node_idx is None:
                if not allow_none:
                    raise IndexError(f"Index {node_idx} is not allowed in this method.")
            else:
                if node_idx < 0 or node_idx >= len(self):
                    raise IndexError(
                        f"Index {node_idx} is out of bounds for this"
                        f" {'sub' if self.is_subtree() else ''}tree with {len(self)} nodes."
                    )

            return func(self, node_idx, *args, **kwargs)

        return wrapper

    if _func is None:
        return decorator  # 2
    else:
        return decorator(_func)


def validate_arrays(parents, descendants):
    assert torch.all(
        descendants == tensortree.descendants_from_parents(parents)
    ), "descendants should be equal to decoded descendants"
    assert torch.all(
        parents == tensortree.parents_from_descendants(descendants)
    ), "parents should be equal to decoded parents"
    assert (descendants < 0).sum() == 0, "no descendant should be below 0"


def linecount(filename: Union[pl.Path, str]) -> int:
    with open(filename, "rb") as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b"\n") for buf in bufgen)


def get_project_root():
    return pl.Path(__file__).parent.parent.parent


def to_torch(some_sequence: Sequence[Any]) -> torch.Tensor:
    if isinstance(some_sequence, torch.Tensor):
        return some_sequence
    elif isinstance(some_sequence, np.ndarray):
        return torch.from_numpy(some_sequence)
    else:
        return torch.tensor(some_sequence)  # may raise additional errors


def is_tensor_type(sequence: Union[torch.Tensor, np.ndarray, Any]) -> bool:
    return isinstance(sequence, (torch.Tensor, np.ndarray))


def to_matmul_compatibility(x: torch.Tensor) -> torch.Tensor:
    """Brings a tensor into a matmul compatible dtype"""
    if x.is_cuda:
        # bmm doesnt work with integers
        return x.float()
    elif x.dtype == torch.bool:
        # mm doesnt work on boolean arrays
        return x.long()
    elif not x.is_floating_point and (x.size(-1) >= torch.iinfo(x.dtype).max):
        # prevent overflow if we multiply uint8
        return x.long()

    return x


def apply_pad_mask_to_higher_dim_tensor(
    x: torch.Tensor, pad_idx: int, pad_mask: torch.Tensor
):
    assert x.ndimension() == (
        pad_mask.ndimension() + 1
    ), "x should have one dimension more than pad_mask"

    if pad_mask.ndimension() == 2:
        x[pad_mask] = pad_idx
        x[pad_mask[:, None, :].expand_as(x)] = pad_idx
    elif pad_mask.ndimension() == 1:
        # then x is 2-dimensional
        x[pad_mask] = pad_idx
        x[:, pad_mask] = pad_idx
    else:
        raise ValueError("pad mask should be either 1d or 2d")

    return x


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    print("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    print("Loading embedding file from", embed_path)
    from tqdm import tqdm

    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in tqdm(f_embed):
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


whitespace_replacemap = {
    " ": "·",
    "\t": "↹",
    "\v": "↦",
    "\n": "⏎",
    "\r": "↵",
}
whitespace_restoremap = {v: k for k, v in whitespace_replacemap.items()}

import re

RE_REPLACE_WHITESPACE = re.compile(
    "|".join(sorted(re.escape(k) for k in whitespace_replacemap))
)
RE_RESTORE_WHITESPACE = re.compile(
    "|".join(sorted(re.escape(k) for k in whitespace_restoremap))
)


def replace_whitespace(text: Union[str, List[str]]):
    """
    Replaces tabs, newlines and spaces with unicode symbols
    """

    def _replace(string: str) -> str:
        if isinstance(string, str):
            return RE_REPLACE_WHITESPACE.sub(
                lambda m: whitespace_replacemap[m.group()], string
            )
        else:
            return string

    if isinstance(text, (list, tuple)):
        res = [_replace(string) for string in text]
        return text.__class__(res)

    return _replace(text)


def restore_whitespace(text: str) -> str:
    return RE_RESTORE_WHITESPACE.sub(lambda m: whitespace_restoremap[m.group()], text)


def parents_from_descendants(descendants: Sequence[int]) -> torch.Tensor:
    """not very performant, but it works."""

    stack_idx = [0]
    stack_open_descendants = [descendants[0]]

    parents = [-1]

    for original_idx, num_descendants in enumerate(descendants[1:], start=1):
        parents.append(stack_idx[-1])

        stack_idx.append(original_idx)
        stack_open_descendants.append(num_descendants + 1)

        stack_open_descendants = [d - 1 for d in stack_open_descendants if (d - 1) > 0]
        stack_idx = stack_idx[: len(stack_open_descendants)]

    return (
        descendants.new_tensor(parents)
        if isinstance(descendants, torch.Tensor)
        else torch.tensor(parents, dtype=torch.long)
    )


def descendants_from_parents(parents: Sequence[int]) -> torch.Tensor:
    descendants = (
        parents.new_zeros(parents.size(-1))
        if isinstance(parents, torch.Tensor)
        else torch.zeros(len(parents), dtype=torch.long)
    )

    active = torch.full_like(
        descendants, fill_value=False, dtype=torch.bool
    )  # bool tensor with all false

    for node_idx, parent_idx in enumerate(parents):
        active[(parent_idx + 1) :] = False  # deactivate closed branch
        descendants[active] += 1  # increment descendants on all active nodes
        active[node_idx] = True  # set current node as active

    return descendants


def descendants_from_node_incidences(node_incidences: torch.Tensor):
    """Computes the descendants array from a (batched) node incidence matrix.

    If the matrix is batched, the returned descendants array will have -1 on padded indices.
    """
    return node_incidences.sum(-2) - 1
