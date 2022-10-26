from collections import deque
import logging
from typing import Optional, Union, Tuple, Generator

import torch
import numpy as np
from numpy.random import Generator as NpGenerator

from cocos import tensortree
from cocos import tokenizer


log = logging.getLogger(__name__)


def sample_node_indices(
    tree: tensortree.TensorTree,
    seed: Union[int, NpGenerator],
    size: int = 1,
    candidate_indices: Optional[np.ndarray] = None,
    mode: str = "descendants",
    replace: bool = True,
    include_leaves: bool = False,
) -> np.ndarray:
    """
    Returns array of n=size sampled node indices.

    :param tree: The tree from which indices should be sampled.
    :param seed: Seed
    :param size: Number of indices to sample.
    :param candidate_indices: Restrict sampling to these indices.
    :param mode: chose from ("uniform", "descendants"). probabilities proportional to number of descendants or
                 uniform for all nodes. In mode "descendants" no leaves are sampled, toggle `include_leaves`
                 to sample leaves.
    :param replace: Sample with replacement (might sample the same index multiple times)
    :param include_leaves: When using mode "descendants" also draw leaves.
    :return:
    """
    seed = np.random.default_rng(seed)

    sample_from_all_nodes = candidate_indices is None

    if not sample_from_all_nodes and candidate_indices.size == 0:
        # print("... No candidates left.")
        return

    if sample_from_all_nodes:
        candidate_indices = len(tree)

    if mode == "uniform":
        probs = None  # default
    elif mode == "descendants":
        descendants = tree.descendants

        # fixme this could actually be dependent of number of leaves
        if sample_from_all_nodes:
            probs = descendants.float().numpy()
        else:
            probs = descendants[candidate_indices].float().numpy()

        if include_leaves:
            probs += 1  # leaves have 0 descendants

        probs /= probs.sum()
    else:
        raise ValueError(f"Sample mode {mode} unknown.")

    if isinstance(size, torch.Tensor):
        size = size.item()

    # print((f"{candidate_indices}, size={size}, replace={replace}"))
    # sample a node that should be removed
    candidate = seed.choice(candidate_indices, size=size, p=probs, replace=replace)
    return candidate


def sample_node_indices_until(
    tree: tensortree.TensorTree,
    max_tree_size: int,
    max_subtree_size: int,
    min_subtree_size: int,
    seed: Union[int, NpGenerator],
    sample_mode: str = "descendants",
    blocked_node_mask: Optional[torch.Tensor] = None,
    dont_sample_whitespace_nodes: bool = False,
) -> Tuple[bool, Optional[torch.Tensor]]:
    """
    Samples nodes until their removal would lead the tree to have less than max_tree_size tokens.
    Will return sampled node_indices even if the tree may be smaller.

    :param tree: The tree
    :param max_tree_size: Amount of tokens left in the tree, before sampling is stopped.
    :param max_subtree_size: Maximum amount of tokens in a sampled subtree (which would then be removed)
    :param min_subtree_size: Minimum amount of tokens in a sampled subtree (which would then be removed)
    :param seed:
    :param sample_mode:  Sampling mode as in `sample_node_indices()`
    :return: Tuple[bool, Tensor]: Boolean indicating if the tree is shortened enough and a tensor with
                                  the sampled indices.
    """
    leaves = tree.leaves_mask()
    num_leaves = leaves.sum()
    num_nonterminals = len(tree) - num_leaves

    # number of tokens
    descendants = tree.descendants

    # true at indices where a branch will be extracted
    branches_to_be_removed = torch.zeros_like(leaves, dtype=torch.bool)

    # indicates which tokens have already been removed
    nodes_to_be_removed = torch.zeros_like(leaves, dtype=torch.bool)

    # we don't know how many leaves a node has. the descendants array includes nonterminals.
    # so we therefore increase max_size by 2
    candidate_mask = (descendants < 2 * max_subtree_size) & (
        min_subtree_size < descendants
    )

    def has_removed_enough_tokens() -> bool:
        # count removed leaves. the removed subtrees will become a new leaf, which we keep
        num_removed_tokens = (
            nodes_to_be_removed & leaves
        ).sum() - branches_to_be_removed.sum()
        # print(f"num_removed_tokens={num_removed_tokens}. Max_tree_size={max_tree_size}. num_leaves={num_leaves}.  (num_leaves - num_removed_tokens)={ (num_leaves - num_removed_tokens)}")
        return (num_leaves - num_removed_tokens) < max_tree_size

    if blocked_node_mask is not None:
        candidate_mask[blocked_node_mask] = False

    candidate_indices = candidate_mask.nonzero().squeeze(-1).numpy()
    if not len(candidate_indices):
        return False, None

    num_samples = min(num_nonterminals, len(candidate_indices))
    candidates = sample_node_indices(
        tree,
        candidate_indices=candidate_indices,
        size=num_samples,
        replace=True,
        mode=sample_mode,
        seed=seed,
    )
    if candidates is None:
        # print(f"Warning. could not sample any candidate. \n"
        #       f"descendants: {descendants}\n"
        #       f"candidate_mask: {candidate_mask}\n"
        #       f"min_subtree_size: {min_subtree_size}\n"
        #       f"max_subtree_size: {max_subtree_size}\n"
        #       f"candidate_indices: {candidate_indices}\n"
        #       f"candidates: {candidates}\n"
        # )
        return False, None

    # sample nodes to remove, but don't remove them yet
    for candidate in candidates:
        if has_removed_enough_tokens():
            break

        if nodes_to_be_removed[candidate] == True:
            # print(f"... candidate is already removed")
            continue

        # sample a node that should be removed
        candidate_end = tree.next_node_not_in_branch(candidate)

        # the amount of tokens the candidate has
        _num_tokens_in_candidate = leaves[candidate:candidate_end].sum()

        if _num_tokens_in_candidate < min_subtree_size:
            # print(f"... Number of tokens ({_num_tokens_in_candidate}) not in requested range [{min_subtree_size}, {max_subtree_size}]")
            continue
        elif _num_tokens_in_candidate > max_subtree_size:
            # print(f"... Number of tokens ({_num_tokens_in_candidate}) not in requested range [{min_subtree_size}, {max_subtree_size}]")
            continue
        elif nodes_to_be_removed[candidate:candidate_end].any():
            # print(f"... Some descendant of candidate is already removed")
            # remove previous candidate and take this candidate
            branches_to_be_removed[candidate:candidate_end] = False
        elif dont_sample_whitespace_nodes and tokenizer.is_whitespace_node(
            tree, candidate
        ):
            continue

        # we can remove this node
        nodes_to_be_removed[candidate:candidate_end] = True
        branches_to_be_removed[candidate] = True

    branch_indices = branches_to_be_removed.nonzero().squeeze(-1)
    # print("branches_to_be_removed", branches_to_be_removed)
    return bool(has_removed_enough_tokens()), branch_indices


def sample_span_indices(
    tree: tensortree.TensorTree,
    min_span_size: int,
    max_span_size: int,
    seed: Union[int, NpGenerator],
    index_or_indices: Optional[Union[torch.Tensor, int, np.ndarray]] = None,
    blocked_node_mask: Optional[torch.Tensor] = None,
) -> list[int]:
    """
    Generates a mask in a tree, by finding a sequence of siblings which can be masked. This may be not possible and
    thus fail. In that case this method tries again with the next index in `index_or_indices` or return None.

    Finding an appropriate window of siblings is done by walking up the tree until we find a node
    that has enough tokens as descendants.

    :param tree: The tree
    :param min_span_size:  Minimum number of tokens in the span.
    :param max_span_size:  Maximum number of tokens in the span.
    :param seed:
    :param index_or_indices: Start walking from these indices. Will return the first for which a span can be found.
                             WIll sample indices if this is None.
    :param blocked_node_mask: Boolean tensor, True at nodes which cant be sampled.

    :return:
    """
    seed = np.random.default_rng(seed)

    if isinstance(index_or_indices, int):
        index_or_indices = [index_or_indices]
    elif index_or_indices is None:
        index_or_indices = sample_node_indices(
            tree=tree, size=len(tree), mode="uniform", seed=seed
        )
        if index_or_indices is None:
            return

    leaf_mask = tree.leaves_mask()

    def can_use_branch_directly(idx):
        return min_span_size < num_tokens_in_branch(idx) < max_span_size

    def has_enough_tokens(idx):
        return min_span_size < num_tokens_in_branch(idx)

    def is_to_large(idx):
        return num_tokens_in_branch(idx) > max_span_size

    def num_tokens_in_branch(idx):
        return leaf_mask[idx : tree.next_node_not_in_branch(idx)].sum()

    # we take the first index and try to obtain between min_span_size
    # and max_span_size tokens by walking up the tree.
    def get_sibling_stacks(
        idx,
    ) -> tuple[list[int], list[int],]:
        left_siblings, right_siblings = [], []

        parent_idx = tree.get_parent(idx)
        # root
        if parent_idx is None:
            return left_siblings, right_siblings

        reached_node = False
        for node in tree.iter_children(parent_idx):
            if node == idx:
                reached_node = True
                continue

            if reached_node:
                right_siblings.append(node)
            else:
                left_siblings.append(node)
        return left_siblings, right_siblings

    def gather_siblings(idx) -> Optional[list[int]]:
        left_siblings, right_siblings = get_sibling_stacks(idx)
        right_siblings.reverse()

        indices_used = deque((idx,))
        sum_tokens = num_tokens_in_branch(idx)

        while (left_siblings or right_siblings) and not (
            min_span_size < sum_tokens < max_span_size
        ):
            stack = None
            left = False

            def use_left_stack():
                nonlocal stack, left
                stack = left_siblings
                left = True

            def use_right_stack():
                nonlocal stack, left
                stack = right_siblings
                left = False

            if left_siblings and right_siblings:
                use_left_stack() if seed.random() < 0.5 else use_right_stack()
            elif left_siblings:
                use_left_stack()
            elif right_siblings:
                use_right_stack()

            sibling_idx = stack.pop()
            if (sum_tokens + num_tokens_in_branch(sibling_idx)) > max_span_size or (
                blocked_node_mask is not None and blocked_node_mask[sibling_idx]
            ):
                # cant use this sibling, so delete stack
                if left:
                    left_siblings = []
                else:
                    right_siblings = []
            else:
                append = indices_used.appendleft if left else indices_used.append
                append(sibling_idx)
                sum_tokens += num_tokens_in_branch(sibling_idx)

        if min_span_size <= sum_tokens <= max_span_size:
            return list(indices_used)

    def walk_upwards(idx) -> Optional[list[int]]:
        if blocked_node_mask is not None and blocked_node_mask[idx]:
            return
        elif can_use_branch_directly(idx):
            return [idx]
        elif is_to_large(idx):
            return
        else:
            parent_idx = tree.get_parent(idx)
            if parent_idx is None:
                return
            if has_enough_tokens(parent_idx):
                # we try to pick a subset of parents siblings, starting from this sibling
                result = gather_siblings(idx)
                if result:
                    return result
            else:
                # repeat with parent
                return walk_upwards(parent_idx)

    for node_idx in index_or_indices:
        gathered_nodes = walk_upwards(node_idx)
        if gathered_nodes:
            return gathered_nodes


def whitespace_nodes(
    tree: tensortree.TensorTree, node_idx: int, siblings: bool = False
) -> Generator[int, None, None]:
    """
    Yields following nodes to node at node_idx, consisting only of whitespace. Looks only for adjacent nodes to
    the right of node_idx.

    :param tree: The tree
    :param node_idx: Node for which adjacent nodes will be returned.
    :param siblings: Restrict to search only whitespace siblings. Otherwise the whitespace node might have a different
                     parent.
    :return:
    """
    if node_idx is None:
        return

    def get_next(i):
        return tree.right_sibling(i) if siblings else tree.next_node_not_in_branch(i)

    # else check if the subtrees in between can be removed as well
    current = get_next(node_idx)
    while tokenizer.is_whitespace_node(tree, current):
        yield current
        current = get_next(current)


def group_nodes(
    tree: tensortree.TensorTree,
    sorted_node_indices: torch.Tensor,
    siblings: bool = False,
) -> Generator[list[Union[torch.Tensor, int]], None, None]:
    """
    Groups nodes and allows whitespace between the nodes.

    :param tree:
    :param sorted_node_indices: Sorted list of node indices to group
    :param siblings: Only group direct siblings. Otherwise groups adjacent nodes.
    :return:
    """
    if len(sorted_node_indices) == 0:
        print("Sorted node indices is empty:", sorted_node_indices)
        return

    first = sorted_node_indices[0]
    indices = [first]

    def get_next(i):
        return tree.right_sibling(i) if siblings else tree.next_node_not_in_branch(i)

    for node_idx in sorted_node_indices[1:]:
        # join if the current node is next to the first node
        if get_next(indices[-1]) != node_idx:
            ws_nodes = list(whitespace_nodes(tree, indices[-1], siblings))
            if ws_nodes and get_next(ws_nodes[-1]) == node_idx:
                for wsp_idx in ws_nodes:
                    indices.append(wsp_idx)
            else:
                yield indices
                indices = []

        indices.append(node_idx)

    if indices:
        yield indices
