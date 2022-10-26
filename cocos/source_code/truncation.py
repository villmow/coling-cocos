from dataclasses import dataclass, field
import logging
import traceback
from typing import Optional, Union, Tuple, Generator

from numpy.random import Generator as NpGenerator

import torch

from cocos import tensortree
from cocos.source_code import sample_node_indices_until, group_nodes, extract_sample
from cocos.source_code import CodeTree, TruncatedCodeTree, ExtractedCodeTree


log = logging.getLogger(__name__)


@dataclass
class TruncateCodeConfig:
    max_folds: int = field(
        default=100, metadata={"help": "Maximum this amount of folds."}
    )

    min_subtree_size: int = 40
    max_subtree_size: int = 1024
    max_tree_size: int = 1024

    fold_smaller_samples: bool = True

    smaller_subtrees_max_tree_size_percentage: float = 0.7
    smaller_subtrees_min_subtree_size_percentage: float = 0.2

    distinct_fold_tokens: bool = field(
        default=False, metadata={"help": "Each fold gets a unique token."}
    )

    join_whitespace: bool = field(
        default=True, metadata={"help": "Join folded nodes separated by whitespace."}
    )
    only_join_siblings: bool = field(
        default=False,
        metadata={"help": "Only join sibling nodes separated by whitespace."},
    )

    sample_mode: str = field(
        default="descendants",
        metadata={
            "help": "Draw candidates for folding uniform, "
            "or depending on the number of descendants."
        },
    )


def truncate_sample(
    sample: CodeTree,
    replacement_tokens,
    distinct_replacement_tokens: bool,
    seed: Union[int, NpGenerator],
    max_tree_size: int,
    max_subtree_size: int,
    min_subtree_size: int,
    sample_mode: str = "descendants",
    join_removed_nodes: bool = True,
    only_join_siblings: bool = False,
) -> Generator[Union[TruncatedCodeTree, ExtractedCodeTree], None, None]:
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
    max_tree_size = max_subtree_size if max_tree_size is None else max_tree_size

    tree_is_small_enough, indices_to_fold = sample_node_indices_until(
        tree=sample.tree,
        max_subtree_size=max_subtree_size,
        min_subtree_size=min_subtree_size,
        max_tree_size=max_tree_size,
        sample_mode=sample_mode,
        seed=seed,
    )

    if indices_to_fold is None or len(indices_to_fold) == 0:
        log.debug(
            f"Could not fold sample ({len(sample.tree)} tokens). {sample.meta.repository}/{sample.meta.path}"
        )
        return

    if join_removed_nodes:
        indices = list(
            group_nodes(sample.tree, indices_to_fold, siblings=only_join_siblings)
        )
        cutout_indices = []
        indices_to_replace = []
        replacement_mask = []

        for index_or_indices in indices:
            cutout_indices.append(
                index_or_indices[0]
            )  # only the first index of every subtree will get a replacement
            indices_to_replace.extend(index_or_indices)
            replacement_mask.append(True)
            replacement_mask.extend(False for _ in range(len(index_or_indices) - 1))

            # if len(index_or_indices) > 1:
            #     for i in range(len(index_or_indices) - 1):
            #         idx1 = index_or_indices[i]
            #         idx2 = index_or_indices[i + 1]
            #         assert sample.tree.next_node_not_in_branch(idx1) == idx2

        num_replacements = len(cutout_indices)
        replacements = (
            replacement_tokens[:num_replacements]
            if distinct_replacement_tokens
            else replacement_tokens[0].expand(num_replacements)
        )
        try:
            truncated_tree, new_cutout_indices = tensortree.delete_nodes(
                sample.tree,
                indices_to_replace,
                replacements,
                replacement_mask,
                return_node_indices=True,
            )
        except RuntimeError as e:
            log.error(f"[truncate_code_sample] Caught exception: {repr(e)}")
            log.error(str(sample))
            log.error(traceback.format_exc())
            return
    else:
        indices = indices_to_fold
        cutout_indices = indices

        num_replacements = len(indices)
        replacements = (
            replacement_tokens[:num_replacements]
            if distinct_replacement_tokens
            else replacement_tokens[0].expand(num_replacements)
        )
        try:
            truncated_tree, new_cutout_indices = tensortree.delete_nodes(
                sample.tree, indices, replacements, return_node_indices=True
            )
        except RuntimeError as e:
            log.error(f"[truncate_code_sample] Caught exception: {repr(e)}")
            log.error(str(sample))
            log.error(traceback.format_exc())
            return

    extracted_samples = (
        extract_sample(sample, node_or_nodes, replacement[None])
        for node_or_nodes, replacement in zip(indices, replacements)
    )

    if min_subtree_size <= truncated_tree.leaves_mask().sum() <= max_tree_size:
        yield TruncatedCodeTree(
            truncated_tree,
            meta=sample.meta,
            cutout_indices=new_cutout_indices,
            old_cutout_indices=cutout_indices,
        )

    yield from extracted_samples


def truncate_code(
    sample: Union[dict, CodeTree],
    seed: Union[int, NpGenerator],
    folding_cfg: TruncateCodeConfig,
    folding_replacement_tokens: Union[list[int], torch.Tensor],
) -> Generator[Union[TruncatedCodeTree, ExtractedCodeTree, CodeTree], None, None]:
    """
    Takes a piece of source_code and returns new pieces of source_code by selecting large enough parts of the source_code as separate
    samples.
    """
    if isinstance(sample, dict):
        sample = CodeTree.from_dict(sample)

    # tree to large, fold and chunk in smaller subtrees
    if sample.size > folding_cfg.max_tree_size:
        yield from truncate_sample(
            sample,
            replacement_tokens=folding_replacement_tokens,
            distinct_replacement_tokens=folding_cfg.distinct_fold_tokens,
            max_subtree_size=folding_cfg.max_subtree_size,
            min_subtree_size=folding_cfg.min_subtree_size,
            max_tree_size=folding_cfg.max_tree_size,
            sample_mode=folding_cfg.sample_mode,
            seed=seed,
            join_removed_nodes=folding_cfg.join_whitespace,
            only_join_siblings=folding_cfg.only_join_siblings,
        )
    # tree can be used as a sample, but we extract smaller samples from it anyways
    else:
        # --> add complete sufficiently large tree
        yield sample

        # create smaller samples
        if (
            folding_cfg.fold_smaller_samples
            and 0 < folding_cfg.smaller_subtrees_max_tree_size_percentage < 1
            and folding_cfg.min_subtree_size < sample.size
        ):
            yield from truncate_sample(
                sample,
                replacement_tokens=folding_replacement_tokens,
                distinct_replacement_tokens=folding_cfg.distinct_fold_tokens,
                max_subtree_size=int(
                    sample.size * folding_cfg.smaller_subtrees_max_tree_size_percentage
                ),
                min_subtree_size=min(
                    folding_cfg.min_subtree_size,
                    int(
                        sample.size
                        * folding_cfg.smaller_subtrees_min_subtree_size_percentage
                    ),
                ),
                max_tree_size=int(
                    sample.size * folding_cfg.smaller_subtrees_max_tree_size_percentage
                ),
                sample_mode=folding_cfg.sample_mode,
                seed=seed,
                join_removed_nodes=folding_cfg.join_whitespace,
                only_join_siblings=folding_cfg.only_join_siblings,
            )
