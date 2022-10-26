from functools import lru_cache
from typing import Optional

import torch

from cocos import tensortree
from cocos import tokenizer
from cocos.source_code import (
    CodeTree,
    TruncatedCodeTree,
    ExtractedCodeTree,
    CodePair,
    CodeTreePair,
)


def num_whitespace_after_newline(
    tree: tensortree.TensorTree, newline_idx: int, search_tabs: bool = False
) -> int:
    if tree.get_node_data(tree.get_parent(newline_idx)) == 5:  # BPE nonterminal
        tokens_after_newline = tree.node_data[newline_idx : tree.step_out(newline_idx)]
    else:
        tokens_after_newline = tree.node_data[
            newline_idx : tree.next_node_not_in_branch(newline_idx)
        ]

    if search_tabs:
        # only check tabs if we didnt find spaces
        return (tokens_after_newline == 7).sum()
    else:
        return tokenizer.NUM_SPACES_IN_TOKEN[tokens_after_newline].sum()


def indentation_level_before_index(
    tree: tensortree.TensorTree, idx: int, search_tabs: bool = False
) -> int:
    def seek_last_newline_idx() -> int:
        for i in range(idx, -1, -1):
            if tokenizer.is_newline(tree.get_node_data(i)):
                return i

    newline_idx = seek_last_newline_idx()
    if newline_idx is None:
        return

    return num_whitespace_after_newline(tree, newline_idx, search_tabs)


@lru_cache(maxsize=128)
def get_indentation_tokens_for_spaces(num_spaces: int):
    tokens = []
    num_spaces = int(num_spaces)
    num_eights = num_spaces // 8
    tokens.extend([11] * num_eights)
    remainder = num_spaces - (num_eights * 8)
    if remainder == 0:
        return torch.tensor(tokens, dtype=torch.long)

    num_fours = remainder // 4
    tokens.extend([12] * num_fours)
    remainder = remainder - num_fours * 4
    if remainder == 0:
        return torch.tensor(tokens, dtype=torch.long)

    num_twos = remainder // 2
    tokens.extend([13] * num_twos)
    num_ones = remainder - num_twos * 2

    if num_ones > 0:
        tokens.extend([14] * num_ones)

    return torch.tensor(tokens, dtype=torch.long)


@lru_cache(maxsize=128)
def get_indentation_tokens_for_tabs(num_tabs: int):
    return torch.tensor([7] * num_tabs, dtype=torch.long)


def dedent(
    tree: tensortree.TensorTree, indentation_level: int, search_tabs: bool = False
):
    if indentation_level is None or indentation_level == 0:
        return tree

    newline_mask = tree.node_data == 10
    newline_mask |= tree.node_data == 9

    node_indices_to_replace = {}
    node_indices = []
    replacements = []
    additional_data = []

    # fixme redo with masks
    # parent_mask = torch.zeros_like(newline_mask, dtype=torch.bool)
    # parents_with_newlines = tree.parents[newline_mask]
    # parent_mask[parents_with_newlines]

    for newline_idx in newline_mask.nonzero():
        newline_idx = newline_idx.squeeze()
        newline_parent = tree.get_parent(newline_idx)

        if tree.get_node_data(newline_parent) != 5:  # bpe token
            continue

        # use rightmost newline_idx
        node_indices_to_replace[newline_parent.item()] = max(
            node_indices_to_replace.get(newline_parent, 0), newline_idx
        )

    # print("masked_idx", masked_idx)
    # print("indentation_level", indentation_level)

    for parent_idx, newline_idx in node_indices_to_replace.items():
        num_newlines_before = newline_idx - parent_idx
        num_whitespace = num_whitespace_after_newline(tree, newline_idx, search_tabs)
        if search_tabs:
            replacement = get_indentation_tokens_for_tabs(
                num_whitespace - indentation_level
            )
        else:
            replacement = get_indentation_tokens_for_spaces(
                num_whitespace - indentation_level
            )

        children = torch.cat(
            [tree[parent_idx].node_data[1 : (num_newlines_before + 1)], replacement]
        )
        node_indices.append(parent_idx)
        replacements.append(children)
        # just_repeat additional data of first child
        additional_data.append(
            [
                add_data.unsqueeze(0).repeat(len(children))
                for add_data in tree[parent_idx].get_additional_node_data(1)
            ]
        )

        # print("#" * 100)
        # print("num_newlines_before", num_newlines_before)
        # print("newline_idx", newline_idx)
        # print("num_s", num_s)
        # print("replacement", replacement)
        # print("tree[parent].node_data", tree[parent_idx].node_data)
        # print("children", children)
        # print("add", additional_data)

    new_tree = tensortree.replace_children_of_nodes(
        tree, node_indices, replacements, additional_data
    )
    return new_tree


def dedent_target(sample: CodeTreePair) -> Optional[CodeTreePair]:
    if not isinstance(sample.target, ExtractedCodeTree):
        raise ValueError(
            "Codepair should consist of TruncatedCodeTree source and ExtractedCodeTree target."
        )
    if not isinstance(sample.source, TruncatedCodeTree):
        raise ValueError(
            "Codepair should consist of TruncatedCodeTree source and ExtractedCodeTree target."
        )

    cut_tree = sample.source.tree
    cutout_tree = sample.target.tree

    masked_idx = sample.target.cutout_index
    indentation_level = indentation_level_before_index(
        cut_tree, masked_idx, search_tabs=False
    )
    search_tabs = False
    if indentation_level is None or indentation_level == 0:
        indentation_level = indentation_level_before_index(
            cut_tree, masked_idx, search_tabs=True
        )
        search_tabs = True

    if indentation_level is None or indentation_level == 0:
        # print("Could not detect indentation level", sample)
        return sample

    dedented_tree = dedent(cutout_tree, indentation_level, search_tabs)

    if dedented_tree is None:
        # print("Could not dedent tree", sample)
        return sample

    return CodeTreePair(
        source=sample.source,
        target=ExtractedCodeTree(
            dedented_tree,
            meta=sample.target.meta,
            cutout_index=sample.target.cutout_index,
            replacement=sample.target.replacement,
        ),
    )
