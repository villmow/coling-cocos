from typing import Optional, Union, Tuple, Generator

import torch

from cocos import tensortree

from cocos.source_code import CodeTree, ExtractedCodeTree, TruncatedCodeTree


def extract_tree(
    tree: tensortree.TensorTree,
    indices: Union[list, int, torch.Tensor],
    new_root_token: Optional[int] = None,
) -> Tuple[int, tensortree.TensorTree]:
    """
    Returns a new tree with the nodes in indices

    :param tree: The tree. It will be not altered!
    :param indices: A new tree from the nodes at indices.
    :return:
    """
    if isinstance(indices, torch.Tensor) and indices.ndim == 0:
        indices = indices[None]

    if len(indices) == 1:
        # single sample tree
        cutout_idx = indices[0]
        subtree = tree[cutout_idx]
    elif len(indices) > 1:
        if new_root_token is None:
            raise ValueError(
                "Trying to concatenate node_indices. Provide token for new root node."
            )

        trees = [tree[idx] for idx in indices]
        cutout_idx = indices[0]

        subtree = tensortree.cat(
            trees,
            new_root_node_data=new_root_token,
            new_root_additional_data=trees[0].get_additional_node_data(0),
        )
    else:
        raise ValueError("indices needs to be iterable")
    return cutout_idx, subtree


def extract_sample(
    sample: CodeTree,
    node_or_nodes: Union[list[int, torch.Tensor], torch.Tensor],
    new_root_token: Optional[int] = None,
) -> ExtractedCodeTree:
    cutout_idx, tree = extract_tree(sample.tree, node_or_nodes, new_root_token)

    return ExtractedCodeTree(
        tree, cutout_index=cutout_idx, meta=sample.meta, replacement=new_root_token
    )


def extract_subtrees(tree: tensortree.TensorTree, indices):
    current_trees = []
    extracted_idx = None
    mask = []  # fixme delete
    for i in range(len(indices)):
        idx = indices[i]
        is_index_to_yield = mask[i]
        current_tree = tree[idx]

        if is_index_to_yield and len(current_trees) == 1:
            # single sample tree
            t = current_trees.pop()

            yield extracted_idx, t
            extracted_idx = idx

        elif is_index_to_yield and len(current_trees) > 1:
            # concatenate trees before
            t = tensortree.cat(
                current_trees,
                new_root_node_data=999,
                new_root_additional_data=current_trees[0].get_additional_node_data(
                    extracted_idx
                ),
            )
            yield extracted_idx, t
            extracted_idx = idx
        elif not current_trees:
            # should not happen
            assert False
        else:
            current_trees.append(current_tree)
            # dont set extracted_idx
