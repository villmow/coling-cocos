from dataclasses import dataclass
from functools import lru_cache
import logging
import random
from typing import Tuple, Union, Optional

import torch
import numpy as np
from numpy.random import Generator as NpGenerator

from cocos import tensortree
from cocos.source_code.dataclasses import CodeTreePair, CodePair, Code, CodeTree


log = logging.getLogger(__name__)


ALL_IDENTIFIER_NODE = "[IDENTIFIER]"
VARIABLE_IDENTIFIERS = {
    "c": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
        },
        "primitive_types": {
            "[primitive_type]",
            "[sized_type_specifier]",
        },
        "methods": {
            "[field_identifier]",
        },  # attributes/methods of objects
    },
    "c-sharp": {
        "identifier": {
            "[identifier]",
        },
        "types": set(),
        "primitive_types": {
            "[predefined_type]",
        },
        "methods": set(),
    },
    "cpp": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
            "[namespace_identifier]",
        },
        "primitive_types": {
            "[primitive_type]",
        },
        "methods": {
            "[field_identifier]",
        },  # actually method names
    },
    "css": {
        "identifier": {
            "[property_name]",
            "[id_name]",
            "[class_name]",
            "[attribute_name]",
        },
        "types": set(),
        "primitive_types": {"[unit]", "[tag_name]"},
        "methods": {
            "[function_name]",
        },
    },
    "go": {
        "identifier": {
            "[identifier]",
            "[field_identifier]",
        },
        "types": {
            "[package_identifier]",
            "[type_identifier]",
        },
        "primitive_types": set(),
        "methods": set(),
    },
    "haskell": {
        "identifier": {
            "[variable]",
            "[constructor]",
        },
        "types": {
            "[type_variable]",
            "[type]",
            "[module]",
        },
        "primitive_types": set(),
        "methods": set(),
    },
    "java": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
        },
        "primitive_types": {
            "[floating_point_type]",
            "[integral_type]",
            "[void_type]",
            "[boolean_type]",
        },
        "methods": set(),
    },
    "javascript": {
        "identifier": {
            "[identifier]",
        },
        "types": set(),
        "primitive_types": set(),
        "methods": {
            "[property_identifier]",
            "[shorthand_property_identifier_pattern]",
            "[shorthand_property_identifier]",
        },  # also properties
    },
    "julia": {
        "identifier": {
            "[identifier]",
        },
        "types": set(),
        "primitive_types": set(),
        "methods": set(),
    },
    "ocaml": {
        "identifier": {
            "[value_name]",
            "[value_pattern]",
            "[class_name]",
            "[constructor_name]",
            "[label_name]",
            "[field_name]",
        },
        "types": {"[module_name]", "[module_type_name]", "[class_type_name]"},
        "primitive_types": {
            "[type_constructor]",
        },
        "methods": {
            "[method_name]",
        },
    },
    "php": {
        "identifier": {
            "[name]",
        },
        "types": set(),
        "primitive_types": {"[primitive_type]"},
        "methods": set(),
    },
    "python": {
        "identifier": {
            "[identifier]",
        },
        "types": set(),
        "primitive_types": set(),
        "methods": set(),
    },
    "ruby": {
        "identifier": {"[identifier]", "[instance_variable]", "[simple_symbol]"},
        "types": {
            "[constant]",
        },
        "primitive_types": set(),
        "methods": set(),
    },
    "rust": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
        },
        "primitive_types": {
            "[primitive_type]",
        },
        "methods": {"[field_identifier]", "[shorthand_field_identifier]"},
    },
    "scala": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
        },
        # "primitive_types": {'[floating_point_type]', '[integral_type]', '[void_type]', '[boolean_type]', },
        "primitive_types": set(),
        "methods": set(),
    },
    "typescript": {
        "identifier": {
            "[identifier]",
        },
        "types": {
            "[type_identifier]",
        },
        "primitive_types": set(),
        "methods": {
            "[property_identifier]",
            "[shorthand_property_identifier_pattern]",
            "[shorthand_property_identifier]",
        },  # also properties
    },
}
LANGUAGES = sorted(list(VARIABLE_IDENTIFIERS.keys()))


@lru_cache(maxsize=30)
def get_identifiers_for_lang(
    lang: str,
    identifier: bool = True,
    types: bool = True,
    primitive_types: bool = True,
    methods: bool = True,
):
    symols = {ALL_IDENTIFIER_NODE}  # always return this nonterminal
    if identifier:
        symols |= VARIABLE_IDENTIFIERS[lang]["identifier"]
    if types:
        symols |= VARIABLE_IDENTIFIERS[lang]["types"]
    if primitive_types:
        symols |= VARIABLE_IDENTIFIERS[lang]["primitive_types"]
    if methods:
        symols |= VARIABLE_IDENTIFIERS[lang]["methods"]
    return symols


def _replace_identifiers_in_string_tree(
    tree: tensortree.TensorTree,
    identifier_nonterminals: set[str],
    replacement_tokens: list = None,
    p: float = 1,
):
    """Works only in trees of strings (no bpe)"""
    raise NotImplementedError(
        "make this work with new version of source_code if needed"
    )

    assert 0 < p <= 1, "p must be between (0,1]"

    identifier_indices = [
        i for i in range(len(tree)) if tree.get_node_data(i) in identifier_nonterminals
    ]

    if replacement_tokens == None:
        replacement_tokens = [f"VAR{i}" for i in range(len(identifier_indices))]

    identifier_mapping = {}

    for idx in identifier_indices:
        assert (
            tree.get_number_of_descendants(idx) == 1
        ), "Identifier should have exactly one descendant"
        identifier = tree.get_node_data(idx + 1)
        if identifier not in identifier_mapping:
            replacement = replacement_tokens[len(identifier_mapping)]
            identifier_mapping[identifier] = replacement

    keys_to_keep = set()
    k = int(len(identifier_mapping) * p)
    if k != len(identifier_mapping):
        keys_to_keep = set(random.sample(identifier_mapping.keys(), k))

    tree = tree.detach()
    for i in identifier_indices:
        identifier = tree.data.node_data[i + 1]

        if identifier not in keys_to_keep:
            tree.data.node_data[i + 1] = identifier_mapping[identifier]

    return tree


def _replace_identifiers_with_bpe(
    tree: tensortree.TensorTree,
    hashes: list,
    replacement_tokens: list[int],
    identifier_nonterminal: int,
    p: float = 1,
):
    """Works only in trees of integers (with bpe)"""
    raise NotImplementedError(
        "make this work with new version of source_code if needed"
    )

    identifier_mask = tree.node_data == identifier_nonterminal
    identifier_indices = identifier_mask.nonzero().squeeze()
    identifier_mapping = {}

    for idx in identifier_indices:
        identifier_hash = hashes[idx]
        if identifier_hash not in identifier_mapping:
            replacement = replacement_tokens[len(identifier_mapping)]
            identifier_mapping[identifier_hash] = {
                "replacement": replacement,
                "node": tree[idx].detach(),
            }
    # sample some identifiers to be removed
    keys_to_keep = set()
    k = int(len(identifier_mapping) * p)
    if k < len(identifier_mapping):
        keys_to_keep = set(random.sample(identifier_mapping.keys(), k))

    # make copy of tree
    tree = tree.detach()

    for idx in torch.flip(identifier_indices, dims=(0,)):
        identifier_hash = hashes[idx]

        if identifier_hash not in keys_to_keep:
            tree = tree.delete_children(
                idx, identifier_mapping[identifier_hash]["replacement"]
            )

    return tree


def get_variable_indices(
    tree: tensortree.TensorTree,
    identifier_nonterminal: int,
    hashes: Optional[torch.Tensor] = None,
):
    identifier_mask = tree.node_data == identifier_nonterminal  # N = Sequence length
    indices_of_nodes = identifier_mask.nonzero().squeeze(
        -1
    )  # I = Number of identifiers

    if hashes is None:
        # take from additional_node_data.
        hashes = tree.additional_data[0][indices_of_nodes].squeeze(-1)

    if not indices_of_nodes.numel():
        raise ValueError("Tree contains no identifiers.")

    unique_hashes, inverse_index, inverse_indices = np.unique(
        hashes.numpy(), return_inverse=True, return_index=True
    )  # UI (unique identifier), I

    return indices_of_nodes, hashes, unique_hashes, inverse_index, inverse_indices


@dataclass
class VariablePositions:
    indices_of_nodes: torch.Tensor  # indices for the original tree
    unique_enumerated: torch.Tensor  # identifiers uniquely enumerated
    inverse_index: torch.Tensor  # maps a unique enumerated token back to original

    def __post_init__(self):
        # pipeline before may return numpy arrays, so convert if thats the case
        attrs = (
            "indices_of_nodes",
            "unique_enumerated",
            "inverse_index",
        )
        for attr in attrs:
            obj = getattr(self, attr)

            if isinstance(obj, np.ndarray):
                obj = torch.from_numpy(obj)

            setattr(self, attr, obj.reshape(-1))

    def select(self, identifier_ids: Union[list[int], np.ndarray, torch.Tensor]):
        """
        Returns a new VariablePositions objects for identifiers contained in identifier_ids.
        The ids in identifier_ids should match with ids in positions.unique_enumerated.
        """
        mask = np.isin(self.unique_enumerated, identifier_ids)
        unique_enumerated = self.unique_enumerated[mask]
        indices_of_nodes = self.indices_of_nodes[mask]
        _, inverse_index, inverse_indices = np.unique(
            unique_enumerated, return_inverse=True, return_index=True
        )
        return VariablePositions(
            indices_of_nodes=indices_of_nodes,  # indices for the original tree
            unique_enumerated=inverse_indices,  # identifiers uniquely enumerated
            inverse_index=inverse_index,  # maps a unique enumerated token back to original
        )


def get_variable_positions(
    tree: tensortree.TensorTree,
    identifier_nonterminal: int,
    hide_prob: float = 1,
    reveal_prob: float = 0,
    seed=42,
) -> Optional[VariablePositions]:
    """
    Works only in trees of integers (with bpe). Returns a dict if return_dict is tr
    """
    seed = np.random.default_rng(seed)

    (
        indices_of_nodes,
        hashes_of_nodes,
        unique_hashes,
        inverse_index,
        inverse_indices,
    ) = get_variable_indices(tree, identifier_nonterminal)

    if hide_prob < 1:
        # randomly remove some hashes
        num_nodes_to_remove = int(len(unique_hashes) * hide_prob + seed.random())
        if num_nodes_to_remove == 0:
            return

        seed.shuffle(unique_hashes)
        hashes_to_mask = unique_hashes[:num_nodes_to_remove]
        mask = np.isin(
            hashes_of_nodes, hashes_to_mask, assume_unique=True, invert=False
        )
        indices_of_nodes = indices_of_nodes[mask]

        # redo it
        inverse_indices = inverse_indices[mask]
        _, inverse_index, inverse_indices = np.unique(
            inverse_indices, return_inverse=True, return_index=True
        )

    # tokens_to_insert = replacement_tokens[inverse_indices]  # I

    if reveal_prob > 0:
        reveal_prob = (len(inverse_indices) * reveal_prob + seed.random()) / len(
            inverse_indices
        )
        keep_hidden_prob = 1 - reveal_prob
        keep_mask = seed.choice(
            [False, True], len(inverse_indices), p=[reveal_prob, keep_hidden_prob]
        )
        keep_mask[inverse_index] = True
        C = (~keep_mask).cumsum(-1)
        i = C[inverse_index]
        inverse_index -= i
        # tokens_to_insert = replacement_tokens[inverse_indices[keep_mask]]  # I
        inverse_indices = inverse_indices[keep_mask]
        indices_of_nodes = indices_of_nodes[keep_mask]

    return VariablePositions(indices_of_nodes, inverse_indices, inverse_index)


def get_unique_variable_positions(
    tree1: tensortree.TensorTree,
    tree2: tensortree.TensorTree,
    identifier_nonterminal: Union[int, torch.Tensor],
    seed: Union[int, NpGenerator],
    percentage_to_keep_in_first: float = 0.5,
    percentage_of_allowed_overlap: float = 0.0,
) -> Tuple[VariablePositions, VariablePositions]:
    seed = np.random.default_rng(seed)
    (
        identifier_indices1,
        id1_hashes,
        unique_hashes1,
        inverse_index1,
        inverse_indices1,
    ) = get_variable_indices(tree1, identifier_nonterminal=identifier_nonterminal)
    (
        identifier_indices2,
        id2_hashes,
        unique_hashes2,
        inverse_index2,
        inverse_indices2,
    ) = get_variable_indices(tree2, identifier_nonterminal=identifier_nonterminal)

    intersection = np.intersect1d(unique_hashes1, unique_hashes2, assume_unique=True)
    if intersection.size == 0:
        raise ValueError("No shared identifiers found.")

    # allow some overlap if configured
    if percentage_of_allowed_overlap > 0:
        assert 0 < percentage_of_allowed_overlap <= 1
        keep_in_intersection_mask = seed.choice(
            [True, False],
            size=len(intersection),
            p=[1 - percentage_of_allowed_overlap, percentage_of_allowed_overlap],
        )
        log.debug(
            f"{len(keep_in_intersection_mask) - keep_in_intersection_mask.sum()} variables overlap"
        )
        intersection = intersection[keep_in_intersection_mask]

    part_of_intersection_to_mask_in_1_mask = seed.choice(
        [True, False],
        size=len(intersection),
        p=[percentage_to_keep_in_first, 1 - percentage_to_keep_in_first],
    )
    hashes_to_mask_in_1 = intersection[part_of_intersection_to_mask_in_1_mask]
    hashes_to_mask_in_2 = intersection[~part_of_intersection_to_mask_in_1_mask]

    mask1 = np.isin(id1_hashes, hashes_to_mask_in_1, assume_unique=False, invert=False)
    indices_to_replace1 = identifier_indices1[mask1]
    indices_to_mask_in_1 = inverse_indices1[mask1]

    mask2 = np.isin(id2_hashes, hashes_to_mask_in_2, assume_unique=False, invert=False)
    indices_to_replace2 = identifier_indices2[mask2]
    indices_to_mask_in_2 = inverse_indices2[mask2]

    # return tokens starting from first token
    _, inverse_index1, inverse_indices1 = np.unique(
        indices_to_mask_in_1, return_inverse=True, return_index=True
    )
    _, inverse_index2, inverse_indices2 = np.unique(
        indices_to_mask_in_2, return_inverse=True, return_index=True
    )

    output1 = VariablePositions(indices_to_replace1, inverse_indices1, inverse_index1)
    output2 = VariablePositions(indices_to_replace2, inverse_indices2, inverse_index2)

    return output1, output2


###################
def mask_identifiers(
    sample: CodeTree,
    identifier_nonterminal: Union[int, torch.Tensor],
    replacement_tokens: torch.Tensor,
    seed: Union[int, NpGenerator],
    hide_prob: float = 1,
    reveal_prob: float = 0,
    return_target: bool = True,
    return_trees: bool = False,
) -> Optional[Union[CodePair, CodeTreePair, Code]]:
    variable_positions = get_variable_positions(
        sample.tree,
        identifier_nonterminal=identifier_nonterminal,
        seed=seed,
        hide_prob=hide_prob,
        reveal_prob=reveal_prob,
    )
    if variable_positions is None:
        return

    if return_trees:
        raise NotImplementedError("Implement this and return CodeTreePair")
    else:
        tokens_to_insert = replacement_tokens[variable_positions.unique_enumerated]
        indices_of_variables = variable_positions.indices_of_nodes
        replaced_leaves, replaced_parts = tensortree.delete_nodes_and_return_leaves(
            sample.tree, indices_of_variables, tokens_to_insert
        )

        code = Code(replaced_leaves, meta=sample.meta)

        if return_target:
            target = construct_target(
                tokens_to_insert, replaced_parts, variable_positions.inverse_index
            )
            return CodePair(code, Code(target, meta=sample.meta))
        else:
            return code


@dataclass
class UniqueVariablesConfig:
    percentage_to_keep_in_first: Optional[float]
    randomize_percentage_to_keep_in_first: bool
    percentage_of_allowed_overlap: Optional[float]


def unique_identifiers_in_pair(
    sample: CodeTreePair,
    cfg: UniqueVariablesConfig,
    identifier_nonterminal: Union[int, torch.Tensor],
    replacement_tokens: torch.Tensor,
    seed: NpGenerator,
    return_trees: bool = False,
) -> Union[CodePair, CodeTreePair]:
    try:
        percentage_to_keep_in_first = cfg.percentage_to_keep_in_first
        if cfg.randomize_percentage_to_keep_in_first:
            percentage_to_keep_in_first = seed.random()

        (
            source_variable_positions,
            target_variable_positions,
        ) = get_unique_variable_positions(
            sample.source.tree,
            sample.target.tree,
            identifier_nonterminal=identifier_nonterminal,
            seed=seed,
            percentage_to_keep_in_first=percentage_to_keep_in_first,
            percentage_of_allowed_overlap=getattr(
                cfg, "percentage_of_allowed_overlap", 0.0
            ),
        )
    except ValueError:
        log.debug("Tree contains no identifiers.")
        if return_trees:
            return sample

        code1 = Code(sample.source.tree.leaves(), meta=sample.source.meta)
        code2 = Code(sample.target.tree.leaves(), meta=sample.target.meta)
        return CodePair(code1, code2)

    if return_trees:
        if source_variable_positions.indices_of_nodes.numel() > 0:
            modified_tree1 = tensortree.delete_nodes(
                sample.source.tree,
                node_indices=source_variable_positions.indices_of_nodes,
                replacements=replacement_tokens[
                    source_variable_positions.unique_enumerated
                ],
            )
            # replaced_leaves1, replaced_parts1 = tensortree.delete_nodes_and_return_leaves(
            #     sample.source.tree,
            #     source_variable_positions.indices_of_nodes,
            #     replacement_tokens[source_variable_positions.unique_enumerated]
            # )  # fixme remove
            # assert torch.equal(
            #     replaced_leaves1, modified_tree1.leaves()
            # ), "both methods should be equal"  # fixme remove

            code1 = CodeTree(modified_tree1, meta=sample.source.meta)
        else:
            code1 = CodeTree(sample.source.tree, meta=sample.source.meta)

        if target_variable_positions.indices_of_nodes.numel() > 0:
            modified_tree2 = tensortree.delete_nodes(
                sample.target.tree,
                node_indices=target_variable_positions.indices_of_nodes,
                replacements=replacement_tokens[
                    target_variable_positions.unique_enumerated
                ],
            )
            # replaced_leaves2, replaced_parts2 = tensortree.delete_nodes_and_return_leaves(
            #     sample.target.tree,
            #     target_variable_positions.indices_of_nodes,
            #     replacement_tokens[target_variable_positions.unique_enumerated]
            # )
            # assert torch.equal(
            #     replaced_leaves2, modified_tree2.leaves()
            # ), "both methods should be equal"  # fixme remove
            code2 = CodeTree(modified_tree2, meta=sample.target.meta)
        else:
            code2 = CodeTree(sample.target.tree, meta=sample.target.meta)

        return CodeTreePair(code1, code2)
    else:
        if source_variable_positions.indices_of_nodes.numel() > 0:
            (
                replaced_leaves1,
                replaced_parts1,
            ) = tensortree.delete_nodes_and_return_leaves(
                sample.source.tree,
                source_variable_positions.indices_of_nodes,
                replacement_tokens[source_variable_positions.unique_enumerated],
            )
            code1 = Code(replaced_leaves1, meta=sample.source.meta)
        else:
            code1 = Code(sample.source.tree.leaves(), meta=sample.source.meta)

        if target_variable_positions.indices_of_nodes.numel() > 0:
            (
                replaced_leaves2,
                replaced_parts2,
            ) = tensortree.delete_nodes_and_return_leaves(
                sample.target.tree,
                target_variable_positions.indices_of_nodes,
                replacement_tokens[target_variable_positions.unique_enumerated],
            )
            code2 = Code(replaced_leaves2, meta=sample.target.meta)
        else:
            code2 = Code(sample.target.tree.leaves(), meta=sample.target.meta)

        return CodePair(code1, code2)


#################
# these should be somewhere else
def construct_target(
    tokens_to_insert: torch.Tensor,
    replaced_parts: list[torch.Tensor],
    inverse_index: np.ndarray,
):
    identifier_to_replacement = (
        (tokens_to_insert[i, None], replaced_parts[i]) for i in inverse_index
    )
    target = torch.cat([x for pair in identifier_to_replacement for x in pair], dim=0)
    return target


def construct_dict(
    tokens_to_insert: torch.Tensor,
    replaced_parts: list[torch.Tensor],
    inverse_index: np.ndarray,
):
    identifier_map = {
        tokens_to_insert[i, None]: replaced_parts[i] for i in inverse_index
    }
    return identifier_map
