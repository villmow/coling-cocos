from dataclasses import dataclass, field
from itertools import zip_longest
import logging
from typing import Optional, Union, Tuple, Generator

import torch
import numpy as np
from numpy.random import Generator as NpGenerator
import scipy.stats as stats

from cocos import tensortree
from cocos.source_code import (
    sample_node_indices,
    sample_span_indices,
    extract_sample,
    sample_node_indices_until,
    CodeTree,
    TruncatedCodeTree,
    CodeTreePair,
    Code,
    CodePair,
)
from cocos.utils import find_runs


log = logging.getLogger(__name__)


def mask_single_span(
    sample: Union[dict, CodeTree],
    seed: Union[int, NpGenerator],
    mask_token: Union[torch.Tensor, int],
    masked_tokens_percentage: Optional[float] = None,  # mask this percentage of tokens
    masked_approximate_num_tokens: Optional[
        float
    ] = None,  # mask this total amount of tokens
    masked_poisson_mean: Optional[float] = None,
    masked_poisson_percentage: Optional[float] = None,
    masked_normal_dist: Optional[
        tuple[float, float]
    ] = None,  # mask this total amount of tokens
    sample_mode: str = "descendants",
) -> Optional[CodeTreePair]:
    """
    Takes a piece of source_code and returns new pieces of source_code.
    """
    seed = np.random.default_rng(seed)

    if isinstance(sample, dict):
        sample = CodeTree.from_dict(sample)

    if (
        sum(
            x is not None
            for x in (
                masked_tokens_percentage,
                masked_approximate_num_tokens,
                masked_poisson_mean,
                masked_poisson_percentage,
                masked_normal_dist,
            )
        )
        != 1
    ):
        log.debug(
            masked_tokens_percentage,
            masked_approximate_num_tokens,
            masked_poisson_mean,
            masked_poisson_percentage,
            masked_normal_dist,
        )
        raise ValueError(
            "Either provide percentage xor absolute number xor poisson mean"
        )

    # add a random number for probabilistic rounding
    if masked_tokens_percentage:
        approximately_masked_tokens = int(
            masked_tokens_percentage * sample.size + seed.random()
        )
    elif masked_approximate_num_tokens:
        approximately_masked_tokens = int(masked_approximate_num_tokens + seed.random())
    elif masked_poisson_mean:
        approximately_masked_tokens = int(seed.poisson(masked_poisson_mean))
    elif masked_poisson_percentage:
        lmb = int(masked_poisson_percentage * sample.size + seed.random())
        approximately_masked_tokens = int(seed.poisson(lmb))
    elif masked_normal_dist:
        mean, std = masked_normal_dist
        lower_bound = sample.size * 0.1
        upper_bound = max(sample.size - 50, sample.size * 0.5)
        dist = stats.truncnorm(
            (lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std
        )
        dist.random_state = seed
        approximately_masked_tokens = dist.rvs()
        if approximately_masked_tokens > sample.size:
            log.info(
                f"approximately_masked_tokens={approximately_masked_tokens}; sample.size={sample.size}"
            )

    min_tokens_to_remove = int(0.8 * approximately_masked_tokens)
    max_tokens_to_remove = int(np.ceil(1.2 * approximately_masked_tokens))

    blocked_node_mask = None
    if isinstance(sample, TruncatedCodeTree):
        blocked_node_mask = get_blocked_node_mask(sample.tree, sample.cutout_indices)
        candidate_indices = (~blocked_node_mask).nonzero().squeeze(-1)

        sampled_indices = sample_node_indices(
            tree=sample.tree,
            size=20,
            mode=sample_mode,
            seed=seed,
            include_leaves=True,
            candidate_indices=candidate_indices,
        )
    else:
        # sample starting indices from which a span will be masked
        sampled_indices = sample_node_indices(
            tree=sample.tree, size=20, mode=sample_mode, seed=seed, include_leaves=True
        )

    if sampled_indices is None:
        # log.debug(f"Could not mask sample. {sample.meta.repository}/{sample.meta.path}")
        # print(sample)
        return

    span_indices_to_mask = sample_span_indices(
        sample.tree,
        min_span_size=min_tokens_to_remove,
        max_span_size=max_tokens_to_remove,
        seed=seed,
        index_or_indices=sampled_indices,
        blocked_node_mask=blocked_node_mask,
    )

    if span_indices_to_mask is None:
        # log.debug(f"Could not mask sample. {sample.meta.repository}/{sample.meta.path}")
        # print(sample)
        return

    span_indices_to_mask = torch.tensor(span_indices_to_mask)

    masked_span = extract_sample(sample, span_indices_to_mask, mask_token)

    masked_tree = sample.tree.delete_siblings(
        span_indices_to_mask, replacement_token=mask_token, dont_check=True
    )

    mask_idx = span_indices_to_mask[0]
    if mask_idx.ndim > 1:
        mask_idx = mask_idx.squeeze()
    if mask_idx.ndim == 0:
        mask_idx = mask_idx.unsqueeze(0)

    source = TruncatedCodeTree(
        masked_tree,
        meta=sample.meta,
        cutout_indices=mask_idx,
        old_cutout_indices=span_indices_to_mask,
    )

    return CodeTreePair(source, masked_span)


@dataclass
class MaskSpanConfig:
    # set only one of the fields below
    masked_tokens_percentage: Optional[float] = field(
        default=None, metadata={"help": "mask this percentage of tokens"}
    )
    masked_approximate_num_tokens: Optional[float] = (
        field(default=None, metadata={"help": "mask this total amount of tokens"}),
    )
    masked_poisson_mean: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": "draw amount of tokens to mask from poisson with this mean!"
            },
        ),
    )
    masked_poisson_percentage: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": "draw amount of tokens to mask from poisson with the mean depending"
                " on the sample size"
            },
        ),
    )
    masked_normal_dist: Optional[tuple[float, float]] = field(
        default=None,
        metadata={
            "help": "draw amount of tokens to mask from truncated normal distribution with following"
        },
    )
    #####################
    sample_mode: str = field(
        default="descendants",
        metadata={
            "help": "Draw candidates for masking uniform, "
            "or depending on the number of descendants."
        },
    )

    def __post_init__(self):
        if (
            sum(
                x is not None
                for x in (
                    self.masked_tokens_percentage,
                    self.masked_approximate_num_tokens,
                    self.masked_poisson_mean,
                    self.masked_poisson_percentage,
                    self.masked_normal_dist,
                )
            )
            != 1
        ):
            raise ValueError(
                "Provide exactly one method to compute number of tokens to mask."
            )


def mask_single_span_cfg(
    sample: Union[dict, CodeTree],
    seed: Union[int, NpGenerator],
    mask_token: Union[torch.Tensor, int],
    cfg: MaskSpanConfig,
) -> Optional[CodeTreePair]:
    return mask_single_span(
        sample,
        mask_token=mask_token,
        seed=seed,
        masked_tokens_percentage=cfg.masked_tokens_percentage,
        masked_approximate_num_tokens=cfg.masked_approximate_num_tokens,
        masked_poisson_mean=cfg.masked_poisson_mean,
        masked_poisson_percentage=cfg.masked_poisson_percentage,
        masked_normal_dist=cfg.masked_normal_dist,
        sample_mode=cfg.sample_mode,
    )


def mask_random(
    sample: Union[CodeTree, Code],
    replacement_tokens: torch.Tensor,
    seed: Union[int, NpGenerator],
    mask_prob: float = 0.15,
) -> Optional[CodePair]:
    tokens = None

    blocked_leaves = None
    if isinstance(sample, CodeTree):
        leaves_mask = sample.tree.leaves_mask()
        tokens = sample.tree.node_data[leaves_mask]

        if isinstance(sample, TruncatedCodeTree):
            blocked_nodes = np.zeros_like(leaves_mask, dtype=bool)
            blocked_nodes[sample.cutout_indices] = True
            blocked_leaves = blocked_nodes[leaves_mask]
            assert len(blocked_leaves) == sample.size

    elif isinstance(sample, Code):
        tokens = sample.data

    if tokens is None:
        raise ValueError(f"{sample.__class__.__name__} is not supported")

    seed = np.random.default_rng(seed)

    # 1 means remove, 0 keep
    mask = seed.choice([False, True], p=[1 - mask_prob, mask_prob], size=(sample.size,))

    if blocked_leaves is not None:
        mask[blocked_leaves] = False  # dont mask these tokens

    source, target = apply_mask_and_construct_target(tokens, mask, replacement_tokens)

    if source is None or target is None:
        return

    return CodePair(
        source=Code(source, meta=sample.meta),
        target=Code(target, meta=sample.meta),
    )


def mask_random_spans(
    sample: Union[CodeTree, Code],
    replacement_tokens: torch.Tensor,
    seed: Union[int, NpGenerator],
    mask_prob: float,
    mean_span_length: int,
) -> Optional[CodePair]:
    tokens = None
    blocked_leaves = None
    if isinstance(sample, CodeTree):
        leaves_mask = sample.tree.leaves_mask()
        tokens = sample.tree.node_data[leaves_mask]

        if isinstance(sample, TruncatedCodeTree):
            blocked_nodes = np.zeros_like(leaves_mask, dtype=bool)
            blocked_nodes[sample.cutout_indices] = True
            blocked_leaves = blocked_nodes[leaves_mask]
            assert len(blocked_leaves) == sample.size

    elif isinstance(sample, Code):
        tokens = sample.data

    if tokens is None:
        raise ValueError(f"{sample.__class__.__name__} is not supported")

    seed = np.random.default_rng(seed)

    # 1 means remove, 0 keep
    mask = np.zeros_like(tokens, dtype=bool)

    approximately_masked_tokens = int(mask_prob * sample.size + seed.random())
    min_tokens_to_remove = int(0.9 * approximately_masked_tokens)
    max_tokens_to_remove = int(np.ceil(1.1 * approximately_masked_tokens))

    approx_num_spans = approximately_masked_tokens / mean_span_length
    approx_num_spans *= 3  # just make sure we sample enough
    approx_num_spans = int(approx_num_spans)

    span_lengths = seed.poisson(mean_span_length, size=approx_num_spans)
    span_positions = seed.permutation(sample.size)

    curr_span = 0
    num_tokens_masked = 0
    for position in span_positions:
        if curr_span == approx_num_spans:
            span_lengths = seed.poisson(
                mean_span_length, size=approx_num_spans
            )  # sample again
            curr_span = 0

        span_length = span_lengths[curr_span]

        try_ = 0  # just to make sure we never get stuck in the loop below
        # skip spans which would mask to many tokens
        while (num_tokens_masked + span_length) > max_tokens_to_remove and try_ < 1000:
            try_ += 1
            curr_span += 1

            # maybe sample spans again if we run out of spans
            if curr_span == approx_num_spans:
                span_lengths = seed.poisson(
                    mean_span_length, size=approx_num_spans
                )  # sample again
                curr_span = 0

            span_length = span_lengths[curr_span]

        start = position
        end = min(position + span_length, sample.size)

        if (num_tokens_masked + span_length) > max_tokens_to_remove:
            assert False

        # free can insert span
        if not mask[start:end].any() and (
            blocked_leaves is None or not blocked_leaves[start:end].any()
        ):
            mask[start:end] = True
            num_tokens_masked += end - start
            curr_span += 1

        if min_tokens_to_remove < num_tokens_masked:
            break  # masked enough

    source, target = apply_mask_and_construct_target(tokens, mask, replacement_tokens)

    if source is None or target is None:
        return

    return CodePair(
        source=Code(source, meta=sample.meta),
        target=Code(target, meta=sample.meta),
    )


def apply_mask_and_construct_target(
    tokens: torch.Tensor, mask: np.ndarray, replacement_tokens: torch.Tensor
):
    """Inserts replacement tokens at the masked positions. Treats consequitive mask tokens as single masks."""

    run_values, run_starts, run_lengths = find_runs(mask)
    slices = tokens.tensor_split(torch.from_numpy(run_starts[1:]))

    kept_parts = slices[::2]
    masked_parts = slices[1::2]

    if run_values[0] == 1:
        kept_parts, masked_parts = masked_parts, kept_parts

    if not masked_parts:
        return None, None

    replacements = replacement_tokens[: len(masked_parts)]

    # and concatenate the separate pieces
    source = torch.cat(
        [
            x.view(-1)
            for pair in zip_longest(kept_parts, replacements)
            for x in pair
            if x is not None
        ]
    )
    target = torch.cat(
        [
            x.view(-1)
            for pair in zip_longest(replacements, masked_parts)
            for x in pair
            if x is not None
        ]
    )
    return source, target


from functools import lru_cache


@lru_cache(maxsize=1)
def calc_quantiles(mean: int, percentage_covered: float):
    lower_prob = (1 - percentage_covered) / 2
    upper_prob = 1 - lower_prob

    from scipy.stats import norm

    lower_quantile = mean - (norm.ppf(upper_prob) * np.sqrt(mean))
    upper_quantile = mean + (mean - lower_quantile)

    lower_quantile = max(lower_quantile, 1)
    return int(np.floor(lower_quantile)), int(np.ceil(upper_quantile))


def mask_trees_in_sample(
    sample: CodeTree,
    replacement_tokens,
    seed: Union[int, NpGenerator],
    mask_prob: float,
    mean_span_length: Optional[int] = None,
    lower_span_bound: Optional[int] = None,
    upper_span_bound: Optional[int] = None,
    sample_mode: str = "descendants",
    # join_removed_nodes: bool = True, only_join_siblings: bool = False
):
    """
    :param sample:
    :param replacement_tokens:
    :param distinct_replacement_tokens:
    :param seed:
    :param max_tree_size: Amount of tokens left in the tree, before sampling is stopped.
    :param max_subtree_size: Maximum amount of tokens in a sampled subtree (which would then be removed)
    :param min_subtree_size: Minimum amount of tokens in a sampled subtree (which would then be removed)
    :param sample_mode:  Sampling mode as in `sample_node_indices()`
    :param join_removed_nodes:
    :param only_join_siblings:
    :return:
    """
    if mean_span_length is None:
        assert lower_span_bound is not None
        assert upper_span_bound is not None
    else:
        assert lower_span_bound is None
        assert upper_span_bound is None
        lower_span_bound, upper_span_bound = calc_quantiles(mean_span_length, 0.75)

    approximately_masked_tokens = int(mask_prob * sample.size + seed.random())
    approx_kept_tokens = sample.size - approximately_masked_tokens

    blocked_node_mask = None
    if isinstance(sample, TruncatedCodeTree):
        blocked_node_mask = get_blocked_node_mask(sample.tree, sample.cutout_indices)

    tree_is_small_enough, indices_to_mask = sample_node_indices_until(
        tree=sample.tree,
        max_subtree_size=upper_span_bound,
        min_subtree_size=lower_span_bound,
        max_tree_size=approx_kept_tokens,
        sample_mode=sample_mode,
        seed=seed,
        blocked_node_mask=blocked_node_mask,
        dont_sample_whitespace_nodes=True,
    )
    if indices_to_mask is None or len(indices_to_mask) <= 1:
        # log.info(f"Could not mask sample ({sample.size} tokens). {sample.meta.repository}/{sample.meta.path}")
        return

    # if join_removed_nodes:
    #     grouped_indices = list(group_nodes(sample.tree, indices_to_mask, siblings=only_join_siblings))
    #     cutout_indices = []
    #     indices_to_replace = []
    #     replacement_mask = []
    #
    #     for index_or_indices in grouped_indices:
    #         cutout_indices.append(index_or_indices[0])  # only the first index of every subtree will get a replacement
    #         indices_to_replace.extend(index_or_indices)
    #         replacement_mask.append(True)
    #         replacement_mask.extend(False for _ in range(len(index_or_indices) - 1))
    #
    #     indices_to_mask = cutout_indices
    #     # need to make tensortree.delete_nodes_and_return_leaves() work with replacement mask
    replacements = replacement_tokens[: len(indices_to_mask)]
    source, targets = tensortree.delete_nodes_and_return_leaves(
        sample.tree, indices_to_mask, replacements
    )
    target = torch.cat(
        [
            x.view(-1)
            for pair in zip_longest(replacements, targets)
            for x in pair
            if x is not None
        ]
    )

    if not tree_is_small_enough:
        num_tokens_masked = len(target) - len(replacements)
        # log.info(f"Masked only {num_tokens_masked / sample.size}% of tokens")
        #
        # from cocos.tokenizer import load_tokenizer
        # t = load_tokenizer()
        # print(t.decode(sample.tree))
        # print(sample)
        # print()

    return CodePair(
        source=Code(source, meta=sample.meta),
        target=Code(target, meta=sample.meta),
    )


def get_blocked_node_mask(
    tree: tensortree.TensorTree, blocked_node_indices: torch.Tensor
):
    # false for nodes which can be candidates.  true  for blocked nodes
    blocked_node_mask = torch.zeros_like(tree.descendants, dtype=torch.bool)

    for idx in blocked_node_indices:
        blocked_node_mask[idx] = True

        for ancestor in tree.iter_ancestors(idx):
            blocked_node_mask[ancestor] = True

    return blocked_node_mask


############################
# random span no tree analysis
def mask_single_span_no_tree_cfg(
    sample: Union[dict, CodeTree],
    seed: Union[int, NpGenerator],
    mask_token: Union[torch.Tensor, int],
    cfg: MaskSpanConfig,
) -> Optional[CodeTreePair]:
    return mask_single_span_no_tree(
        sample,
        mask_token=mask_token,
        seed=seed,
        masked_tokens_percentage=cfg.masked_tokens_percentage,
        masked_approximate_num_tokens=cfg.masked_approximate_num_tokens,
        masked_poisson_mean=cfg.masked_poisson_mean,
        masked_poisson_percentage=cfg.masked_poisson_percentage,
        masked_normal_dist=cfg.masked_normal_dist,
    )


def mask_single_span_no_tree(
    sample: Union[dict, CodeTree],
    seed: Union[int, NpGenerator],
    mask_token: Union[torch.Tensor, int],
    masked_tokens_percentage: Optional[float] = None,  # mask this percentage of tokens
    masked_approximate_num_tokens: Optional[
        float
    ] = None,  # mask this total amount of tokens
    masked_poisson_mean: Optional[float] = None,
    masked_poisson_percentage: Optional[float] = None,
    masked_normal_dist: Optional[
        tuple[float, float]
    ] = None,  # mask this total amount of tokens
) -> Optional[CodePair]:
    """
    Takes a piece of source_code and returns new pieces of source_code.
    """
    seed = np.random.default_rng(seed)

    if isinstance(sample, dict):
        sample = CodeTree.from_dict(sample)

    if (
        sum(
            x is not None
            for x in (
                masked_tokens_percentage,
                masked_approximate_num_tokens,
                masked_poisson_mean,
                masked_poisson_percentage,
                masked_normal_dist,
            )
        )
        != 1
    ):
        log.debug(
            masked_tokens_percentage,
            masked_approximate_num_tokens,
            masked_poisson_mean,
            masked_poisson_percentage,
            masked_normal_dist,
        )
        raise ValueError(
            "Either provide percentage xor absolute number xor poisson mean"
        )

    # add a random number for probabilistic rounding
    if masked_tokens_percentage:
        approximately_masked_tokens = int(
            masked_tokens_percentage * sample.size + seed.random()
        )
    elif masked_approximate_num_tokens:
        approximately_masked_tokens = int(masked_approximate_num_tokens + seed.random())
    elif masked_poisson_mean:
        approximately_masked_tokens = int(seed.poisson(masked_poisson_mean))
    elif masked_poisson_percentage:
        lmb = int(masked_poisson_percentage * sample.size + seed.random())
        approximately_masked_tokens = int(seed.poisson(lmb))
    elif masked_normal_dist:
        mean, std = masked_normal_dist
        lower_bound = sample.size * 0.1
        upper_bound = max(sample.size - 50, sample.size * 0.5)
        dist = stats.truncnorm(
            (lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std
        )
        dist.random_state = seed
        approximately_masked_tokens = dist.rvs()
        if approximately_masked_tokens > sample.size:
            log.info(
                f"approximately_masked_tokens={approximately_masked_tokens}; sample.size={sample.size}"
            )

    approximately_masked_tokens = int(approximately_masked_tokens)
    tokens = sample.data.leaves()

    if approximately_masked_tokens > len(tokens):
        raise ValueError
    elif approximately_masked_tokens == len(tokens):
        approximately_masked_tokens -= 1  # dont mask everything

    if not approximately_masked_tokens:
        return

    span_start_idx = seed.integers(len(tokens) - approximately_masked_tokens)
    span_end_idx = span_start_idx + approximately_masked_tokens

    source_tokens = torch.cat(
        [tokens[:span_start_idx], torch.tensor([mask_token]), tokens[span_end_idx:]]
    )
    target_tokens = tokens[span_start_idx:span_end_idx]

    source = Code(
        source_tokens,
        meta=sample.meta,
    )
    target = Code(
        target_tokens,
        meta=sample.meta,
    )

    return CodePair(source, target)
