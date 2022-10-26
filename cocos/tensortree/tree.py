from dataclasses import dataclass, field
from typing import (
    Sequence,
    Any,
    Union,
    List,
    Optional,
    Tuple,
    Generator,
    Callable,
    Literal,
)

import numpy as np
import torch

from cocos import tensortree
from cocos.tensortree.render import Style, ContRoundStyle, format_tree
from cocos.tensortree.utils import (
    to_torch,
    validate_index,
    replace_whitespace,
    validate_arrays,
    DEBUG,
)

# Define a type alias for the content of the node sequence
LabelType = Any
TensorType = Union[Sequence[int], np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class TreeStorage:
    """
    Stores a tree with data.
    """

    # either parents or descendants may be None.
    # other sequence types will be converted to tensors.
    parents: TensorType = None
    descendants: Union[torch.Tensor, Sequence[int]] = None

    # some operations (swapping) only work when node_data is a torch.Tensor
    node_data: Union[torch.Tensor, Sequence[LabelType]] = None

    # additional tensors to store
    additional_data: Optional[List[Union[torch.Tensor, Sequence[LabelType]]]] = field(
        default_factory=list
    )

    format: Literal["torch", "numpy"] = "torch"

    def __post_init__(self):
        if self.parents is None and self.descendants is None:
            raise ValueError("Either parents or descendants must be passed")

        if self.parents is None:  # compute parents from descendants
            descendants: torch.Tensor = to_torch(self.descendants).long()
            parents = tensortree.parents_from_descendants(self.descendants)

        elif self.descendants is None:  # compute descendants from parents
            parents: torch.Tensor = to_torch(self.parents).long()
            descendants = tensortree.descendants_from_parents(parents)

        else:  # convert everything to a tensor
            parents: torch.Tensor = to_torch(self.parents).long()
            descendants: torch.Tensor = to_torch(self.descendants).long()

        # node_data may be nothing, in that case simply enumerate the nodes
        if self.node_data is None:
            node_data = torch.arange(len(descendants)).to(descendants)
            # FIXME make node_data optional
        else:
            # node_data is a sequence of strings (tensor incompatible)
            try:
                node_data: torch.Tensor = to_torch(self.node_data).long()
            except (ValueError, TypeError, RuntimeError):
                node_data: List[LabelType] = list(self.node_data)

        if descendants.numel() != len(node_data) != parents.numel():
            raise ValueError(
                f"All arrays need to be of same length and not ({descendants.numel()}, {len(node_data)}, {parents.numel()})."
            )

        if self.additional_data is None:
            object.__setattr__(self, "additional_data", [])

        additional_data = []
        for element in self.additional_data:
            if type(element) is not type(node_data):
                element = to_torch(element)
            if len(node_data) != len(element):
                raise ValueError(
                    "additional data must have same shape and type as node data"
                )
            additional_data.append(element)

        object.__setattr__(self, "additional_data", additional_data)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "descendants", descendants)
        object.__setattr__(self, "node_data", node_data)

        # pretty expensive
        if DEBUG:
            validate_arrays(parents, descendants)


def tree(
    parents: Optional[TensorType] = None,
    node_data: Optional[Sequence[LabelType]] = None,
    descendants: Optional[TensorType] = None,
    additional_data: Optional[List[Sequence[int]]] = None,
):
    """Constructor to build a tree."""

    return TensorTree.from_array(
        parents=parents,
        descendants=descendants,
        node_data=node_data,
        additional_data=additional_data,
    )


class TensorTree:
    @classmethod
    def from_array(
        cls,
        parents: Optional[TensorType] = None,
        descendants: Optional[TensorType] = None,
        node_data: Optional[Sequence[LabelType]] = None,
        additional_data: Optional[List[Sequence[int]]] = None,
    ):
        """Obtain a tree from arrays. A tree can be either defined by a parents or a descendants tensor.
        Additional nodes list can be passed if it contains a string, it will be used for rendering.
        """
        return cls(TreeStorage(parents, descendants, node_data, additional_data))

    @classmethod
    def from_node_incidences(
        cls,
        node_incidences: torch.Tensor,
        node_data: Optional[Sequence[LabelType]] = None,
        additional_data: Optional[List[Sequence[int]]] = None,
    ):
        descendants = tensortree.descendants_from_node_incidences(node_incidences)
        return cls.from_array(
            descendants=descendants,
            node_data=node_data,
            additional_data=additional_data,
        )

    def __init__(self, data: TreeStorage, root_idx: int = 0):
        """
        Initialize with a pointer to tree storage.
        """
        self.data = data
        self.root_idx = root_idx

        if len(self.data.parents) > 0 and (
            self.data.parents[0] != -1
        ):  # and self.data.parents[0] != 0):
            raise ValueError("Parents array seems to have wrong format.")

        self.__len = self.data.descendants.shape[-1]  # cache this

        # span in original array
        self.end = len(self)

        if self.is_subtree():
            self.__len = self.data.descendants[root_idx] + 1
            self.end = root_idx + len(self)

    def __len__(self):
        """The number of nodes in this tree."""
        return self.__len

    def __str__(self):
        return self.pformat(max_nodes=10)

    @validate_index(allow_none=True)
    def __getitem__(
        self, node_idx: Union[int, torch.Tensor, np.ndarray, Tuple[int, ...], None]
    ):
        """Will returns a view of the node at node_idx"""
        if node_idx is None and node_idx != self.root_idx:
            return TensorTree(self.data)  # take me back to the very root

        node_idx = self._to_global_idx(node_idx)

        if node_idx == self.root_idx:
            return self

        return TensorTree(self.data, node_idx)

    def detach(self):
        """Returns a new tree rooted at self.root_idx"""
        from copy import deepcopy

        if isinstance(self.node_data, torch.Tensor):
            new_node_data = self.node_data.clone()
            new_additional_data = [data.clone() for data in self.additional_data]
        else:
            new_node_data = deepcopy(self.node_data)
            new_additional_data = [deepcopy(data) for data in self.additional_data]

        return self.from_array(
            parents=self.parents.clone(),
            descendants=self.descendants.clone(),
            node_data=new_node_data,
            additional_data=new_additional_data,
        )

    def is_subtree(self) -> bool:
        """
        Is this tree instance part of a bigger tree?

        You can retrieve the original tree using
        >>> root = subtree[None]
        """

        return self.root_idx > 0

    def is_descendant_of(
        self, node_ancestor_idx: int, node_descendant_idx: int
    ) -> bool:
        """
        Is one node a descendant of another node?

        :param node_ancestor_idx: The node closer to the root
        :param node_descendant_idx: The descendant
        :return:
        """
        return (
            node_ancestor_idx
            < node_descendant_idx
            <= (node_ancestor_idx + self.get_number_of_descendants(node_ancestor_idx))
        )

    # helpers for individual nodes
    @validate_index
    def get_node_data(self, node_idx: Union[int, torch.Tensor]) -> Any:
        """
        Returns the data of a node.

        :param node_idx: The index of the node.
        """
        node_idx = self._to_global_idx(node_idx)
        return self.data.node_data[node_idx]

    @validate_index
    def get_number_of_descendants(
        self, node_idx: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns the amount of descendants of a node.

        :param node_idx:
        :return:
        """
        node_idx = self._to_global_idx(node_idx)
        return self.data.descendants[node_idx]

    @validate_index
    def get_parent(self, node_idx: Union[int, torch.Tensor]) -> Optional[int]:
        """Returns the parent idx for this node or None if node_idx is root.

        :param node_idx: Node_idx refers to the index of the node in the original tree.
        :return:
        """
        node_idx = self._to_global_idx(node_idx)

        # return None for root
        if node_idx == self.root_idx:
            return

        # FIXME test this! not sure if correct
        old_parent = self.data.parents[node_idx]
        new_parent = old_parent - self.root_idx

        assert new_parent >= -1
        return new_parent

    @validate_index
    def get_additional_node_data(self, node_idx: Union[int, torch.Tensor]) -> List[Any]:
        """
        Returns the additional data of a node.

        :param node_idx: The index of the node.
        """
        node_idx = self._to_global_idx(node_idx)
        return [data[node_idx] for data in self.data.additional_data]

    @property
    def descendants(self) -> torch.Tensor:
        """
        Returns the relevant subset of descendants for this subtree. This will return
         a slice of the data. Changing this object will change the storage and may
         invalidate the tree. This is not checked.
        """
        if self.root_idx == 0:
            return self.data.descendants  # return full array
        else:
            return self.data.descendants[self.root_idx : self.end]  # return slice

    @property
    def node_data(self) -> Sequence[LabelType]:
        """
        Returns the relevant subset of node_data for this tree. This will return
         a slice of the data. Changing this object will change the storage and may
         invalidate the tree. This is not checked.

        """
        if self.root_idx == 0:
            return self.data.node_data
        else:
            return self.data.node_data[self.root_idx : self.end]

    @property
    def parents(self) -> torch.Tensor:
        """
        Returns the relevant subset of parents for this subtree. This returns the underlying
         storage only if it is root. Otherwise will create a new array for this subtree.

        """
        if self.root_idx == 0:
            return self.data.parents

        # compute parents
        if getattr(self, "__subtree_parents", None) is None:
            parents = self.data.parents[self.root_idx : self.end] - self.root_idx
            parents[0] = -1
            self.__subtree_parents = parents  # cache parents for this subtree

        return self.__subtree_parents

    @property
    def additional_data(self) -> List[Sequence[LabelType]]:
        """
        Returns the relevant subset of additional data for this tree. This will return
         a slice of the data. Changing this object will change the storage and may
         invalidate the tree. This is not checked.

        """
        if self.root_idx == 0:
            return self.data.additional_data
        else:
            return [
                data[self.root_idx : self.end] for data in self.data.additional_data
            ]

    def node_incidence_matrix(self) -> torch.BoolTensor:
        """Returns the node incidence matrix for this tree"""
        return tensortree.node_incidence_matrix(self.descendants)

    def adjacency_matrix(
        self, directed: bool = True, direction_up: bool = False, loop_root: bool = False
    ) -> torch.BoolTensor:
        """
         Returns the adjacency matrix for this tree.

        :param directed: Should the adjacency matrix be directed?
        :param direction_up: Per default the root points at its children.
        :param loop_root: Should root have a loop.
        :return:
        """
        return tensortree.adjacency_matrix(
            parents=self.parents,
            directed=directed,
            direction_up=direction_up,
            loop_root=loop_root,
        )

    def ancestral_matrix(self) -> torch.LongTensor:
        """
        Computes the level of the least common ancestor between any node pair
        (or in other words the length of the common prefix between two nodes).

        For this tree:
            0
            ├──  1
            └──  2
                ├──  3
                └──  4
                    └──  5

        With the following node incidence matrix:

        tensor([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

        This method will return:

        tensor([[0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 1, 2, 1, 1],
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 3]])

        Pytorch supports matrix multiplications on cuda tensors only with float tensors,
         thus this method will return a FloatTensor when the node incidence matrix is located
         on cuda. Otherwise it tries to keep the original dtype when possible, otherwise
         returns a LongTensor.

        :return:
        """

        return tensortree.ancestral_matrix(self.node_incidence_matrix())

    def movements(self) -> torch.LongTensor:
        """
        Computes movements between nodes.

        For this tree:
            0
            ├──  1
            └──  2
                ├──  3
                └──  4
                    └──  5

        With the following node incidence matrix:
        tensor([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

        Will return the following movements between nodes.
        tensor([[0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0],
                [2, 2, 1, 0, 1, 1],
                [2, 2, 1, 1, 0, 0],
                [3, 3, 2, 2, 1, 0]])

        You can read it the following:
        To walk from node i to node j one needs to make
         1. res[i, j] upward movements
         2. res[j, i] downward movements

        Some examples:
         - from 0 to 4 -> res[0,4]=0; res[4,0]=2 (0 steps up, 2 steps down)
         - from 1 to 5 -> res[1,5]=1; res[5,1]=3 (1 step  up, 3 steps down)

        Will always return a LongTensor.
        """
        return tensortree.movements(self.node_incidence_matrix())

    def distances(self) -> torch.LongTensor:
        """
        Computes distances between nodes when the tree is seen as undirected.

        For this tree:
            0
            ├──  1
            └──  2
                ├──  3
                └──  4
                    └──  5

        With the following node incidence matrix:
        tensor([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

        Will return the following distances between nodes.
        tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 3, 3, 4],
                [1, 2, 0, 1, 1, 2],
                [2, 3, 1, 0, 2, 3],
                [2, 3, 1, 2, 0, 1],
                [3, 4, 2, 3, 1, 0]])

        Will always return a LongTensor.
        """
        return tensortree.distances(self.node_incidence_matrix())

    def least_common_ancestors(self):
        """
        Return a least common ancestor matrix for this tree.

        :return:
        """
        return tensortree.least_common_ancestors(self.node_incidence_matrix())

    def levels(self) -> torch.LongTensor:
        """
        Computes the level of each node (i.e. the number of edges from a node to the root).
        Root has level 0.

        For this tree:
            0
            ├──  1
            └──  2
                ├──  3
                └──  4
                    └──  5

        With the following node incidence matrix:

        tensor([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1]], dtype=torch.uint8)

        Will return:

        tensor([0, 1, 1, 2, 2, 3])
        """
        return tensortree.levels(self.node_incidence_matrix())

    @validate_index
    def next_node_not_in_branch(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Return the next node, that is not part of the branch of node at node_idx.

        Can be a sibling or any other node that follows after this branch.

        :param node_idx: Can be either an integer or a single value tensor
        :return:
        """
        next_node_idx = node_idx + self.get_number_of_descendants(node_idx) + 1
        return next_node_idx if next_node_idx < len(self) else None

    @validate_index
    def step_out(self, node_idx: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Return the next node, that is not part of the subtree of node_idx's parent.

        Can be a sibling to nodes parent or any other node that follows.

        :param node_idx: Can be either an integer or a single value tensor
        :return:
        """
        nodes_parent = self.get_parent(node_idx)
        if nodes_parent is None:
            return

        return self.next_node_not_in_branch(nodes_parent)

    def node_idx_for_tree_position(
        self, tree_position: Tuple[int, ...]
    ) -> Union[int, torch.Tensor]:
        if not isinstance(tree_position, tuple):
            raise ValueError("tree_position must be tuple")

        def get_nth_child(node_idx, n):
            for i, child_idx in enumerate(self.iter_children(node_idx)):
                if i == n:
                    return child_idx
            raise IndexError

        current_node_idx = 0
        for child_number in tree_position:
            current_node_idx = get_nth_child(current_node_idx, child_number)

        return current_node_idx

    def heights(self):
        """Depth of each subtree"""
        heights = torch.zeros_like(self.descendants)

        for i, layer_mask in enumerate(
            tensortree.mask_layer(self.node_incidence_matrix())
        ):
            heights[layer_mask] = i

        return heights

    # subtree mask
    @validate_index
    def subtree_mask(self, node_idx: Union[int, torch.Tensor]) -> torch.BoolTensor:
        """returns a mask which selects the relevant nodes of a subtree from the array"""

        start = node_idx
        end = self.next_node_not_in_branch(node_idx)

        mask = torch.zeros_like(self.descendants, dtype=torch.bool)
        mask[start:end] = True
        return mask

    def subtree_masks(self) -> torch.BoolTensor:
        """returns a mask for all subtrees"""
        S = len(self)
        positions = torch.arange(end=S)
        descendants = self.descendants + 1
        # every node is on its own path to root
        reps = torch.zeros(3 * S, dtype=torch.int)
        # num repetitions of pad before diag
        reps[::3] = positions
        # num repetitions of position after diag
        reps[1::3] = descendants
        # num repetitions of pad after diag
        reps[2::3] = (descendants - S).abs() - positions

        val_mask = torch.zeros_like(reps, dtype=torch.bool)
        val_mask[1::3] = True

        return torch.repeat_interleave(val_mask.view(-1), reps.view(-1)).view(S, S)

    # leaves
    @validate_index
    def is_leaf(self, node_idx: Union[int, torch.Tensor]) -> bool:
        return self.get_number_of_descendants(node_idx) == 0

    def leaves_mask(self) -> torch.BoolTensor:
        """Returns a boolean mask for all leaf nodes in this tree"""
        # fixme depricate?
        return self.leaf_mask()

    def leaf_mask(self) -> torch.BoolTensor:
        """Returns a boolean mask for all leaf nodes in this tree"""
        return self.descendants == 0

    def leaf_indices(self) -> torch.Tensor:
        return self.leaves_mask().nonzero(as_tuple=False).squeeze(-1)

    def leaves(self) -> Union[torch.Tensor, list]:
        if isinstance(self.node_data, torch.Tensor):
            return self.node_data[self.leaves_mask()]
        else:
            return [self.get_node_data(i) for i in self.leaf_indices()]

    # nonterminals
    @validate_index
    def is_nonterminal(self, node_idx: Union[int, torch.Tensor]) -> bool:
        return self.get_number_of_descendants(node_idx) != 0

    def nonterminal_mask(self) -> torch.BoolTensor:
        """Returns a boolean mask for all leaf nodes in this tree"""
        return self.descendants != 0

    def nonterminal_indices(self) -> torch.Tensor:
        return self.nonterminal_mask().nonzero(as_tuple=False).squeeze(-1)

    def nonterminals(self) -> Union[torch.Tensor, list]:
        """Gets data for all nonterminals."""
        if isinstance(self.node_data, torch.Tensor):
            return self.node_data[self.nonterminal_mask()]
        else:
            return [self.get_node_data(i) for i in self.nonterminal_indices()]

    # children
    @validate_index
    def iter_children(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Generator[torch.Tensor, None, None]:
        """
        Iters over the children of a node with a specific index in a tree.

        :param node_idx: Node to iterate over the children
        """
        branch_end = self.next_node_not_in_branch(
            node_idx
        )  # end of subtree of node_idx

        if branch_end is None:
            branch_end = len(self)

        # next child is at the next position in the descendants array
        next_child = node_idx + 1

        # are we still in the subtree
        while next_child is not None and next_child < branch_end:
            yield next_child
            next_child = self.next_node_not_in_branch(next_child)

    @validate_index
    def children_mask(self, node_idx: Union[int, torch.Tensor]) -> torch.BoolTensor:
        """
        Returns a mask over the child indices of a node.

        :param node_idx: Node to get the children
        """
        return self.adjacency_matrix()[node_idx]

    @validate_index
    def children(self, node_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Returns children indices of a node at a specific index in a tree.

        :param node_idx: Node to get the children
        """
        return self.children_mask(node_idx).nonzero(as_tuple=False).squeeze(-1)

    @validate_index
    def index_of_child_with_data(
        self, node_idx: Union[int, torch.Tensor], data: Any
    ) -> Generator[torch.Tensor, None, None]:
        """
        Returns the first index of a child with a specific `node_data`.
        """
        assert data is not None

        for child_idx in self.iter_children(node_idx):
            if self.get_node_data(child_idx) == data:
                return child_idx

        raise ValueError(f"{data} not in children")

    # siblings
    @validate_index
    def iter_siblings(
        self,
        node_idx: Union[int, torch.Tensor],
        include_left_siblings: bool = True,
        include_right_siblings: bool = True,
    ) -> Generator[torch.Tensor, None, None]:
        """Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set include_right_siblings to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")

        parent_idx = self.get_parent(node_idx)

        # root
        if parent_idx is None:
            return

        reached_node = False
        for node in self.iter_children(parent_idx):
            if node == node_idx:
                reached_node = True
                continue

            if include_left_siblings and not reached_node:
                yield node

            if include_right_siblings and reached_node:
                yield node

    @validate_index
    def right_sibling(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        next_node = self.next_node_not_in_branch(node_idx)
        if next_node is not None:
            if self.get_parent(next_node) == self.get_parent(node_idx):
                return next_node

    @validate_index
    def left_sibling(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if node_idx == 0:
            return

        node_before = node_idx - 1
        parent = self.get_parent(node_idx)

        if node_before == parent:
            return  # no left sibling

        while self.get_parent(node_before) != parent:
            node_before = self.get_parent(node_before)

        return (
            torch.tensor(node_before) if isinstance(node_before, int) else node_before
        )

    @validate_index
    def are_siblings(
        self, node1: Union[int, torch.Tensor], node2: Union[int, torch.Tensor]
    ) -> bool:
        return self.get_parent(node1) == self.get_parent(node2)

    @validate_index
    def siblings_mask(
        self,
        node_idx: Union[int, torch.Tensor],
        include_left_siblings: bool = True,
        include_right_siblings: bool = True,
    ) -> torch.BoolTensor:
        """Node indices with the same parent. The node at node_idx is excluded.

        set include_left_siblings to False to only output siblings which are to the right of node at node_idx.
        set right to False  to only output siblings which are to the left of node at node_idx.
        """
        if not (include_left_siblings or include_right_siblings):
            raise ValueError("Left or right must be specified.")

        parent_idx = self.get_parent(node_idx)

        # root
        if parent_idx is None:
            return self.data.descendants.new_zeros(len(self)).bool()

        all_siblings = self.children_mask(parent_idx)

        assert all_siblings[node_idx], "this should be true"
        # exlude node at node_idx
        all_siblings[node_idx] = False

        # exclude everything before node_idx
        if not include_left_siblings:
            all_siblings[:node_idx] = False

        # exclude everything after node_idx
        if not include_right_siblings:
            all_siblings[node_idx + 1 :] = False

        return all_siblings

    def sibling_masks(self) -> torch.BoolTensor:
        """A mask with every sibling for every node in the tree."""
        return self.parents[None, :] == self.parents[:, None]

    @validate_index
    def siblings(
        self,
        node_idx: Union[int, torch.Tensor],
        check_left: bool = True,
        check_right: bool = True,
    ) -> torch.Tensor:
        return (
            self.siblings_mask(node_idx, check_left, check_right)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )

    @validate_index
    def has_sibling(
        self,
        node_idx: Union[int, torch.Tensor],
        check_left: bool = True,
        check_right: bool = True,
    ) -> bool:
        """
        Is there a sibling to this node?

        :param node_idx:
        :param check_left: look left for siblings (set to False to only check for right siblings)
        :param check_right: look right for siblings (set to False to only check for left siblings)
        :return:
        """
        node_parent = self.get_parent(node_idx)
        if node_parent is None:
            return False

        result = False
        if check_right:
            next_node = self.next_node_not_in_branch(node_idx)
            if next_node is not None:
                next_nodes_parent = self.get_parent(next_node)
                has_right_sibling = next_nodes_parent == node_parent
                result = result or has_right_sibling
        if check_left and not result:
            # otherwise something should be between parent and node
            has_left_sibling = (node_parent + 1) != node_idx
            result = result or has_left_sibling

        return result

    # ancestors
    @validate_index
    def iter_ancestors(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Generator[torch.Tensor, None, None]:
        node_parent = self.get_parent(node_idx)
        num_iterations = 0
        max_steps = len(self)
        while node_parent is not None:
            yield node_parent
            node_parent = self.get_parent(node_parent)
            num_iterations += 1

            if num_iterations > max_steps:
                raise ValueError("Infinite loop detected.")

    # pretty printing
    def pformat(
        self,
        max_nodes: Optional[int] = None,
        node_renderer: Callable[[Tuple[int, Any]], str] = None,
        style: Union[Style] = ContRoundStyle,
    ) -> str:
        """
        Pretty prints a tree up to `max_nodes`. Define a node_renderer for custom node types (e.g. Dictionaries).
        :param max_nodes: Render up to this amount of nodes.
        :param node_renderer: A function that outputs a string.
        :param style: Style the tree.
        :return:
        """

        def replace_ws(node_idx, node):
            return replace_whitespace(node)

        if node_renderer is None:
            node_renderer = replace_ws

        return format_tree(
            tree=self, max_nodes=max_nodes, node_renderer=node_renderer, style=style
        )

    def pprint(
        self,
        max_nodes: Optional[int] = None,
        node_renderer: Callable[[Tuple[int, Any]], str] = None,
        style: Union[Style] = ContRoundStyle,
    ):
        """See pformat for description of arguments."""
        print("TensorTree():\n", self.pformat(max_nodes, node_renderer, style))

    @validate_index
    def _to_global_idx(
        self, node_idx: Union[int, torch.Tensor]
    ) -> Union[int, torch.Tensor]:
        """Transfers a node_idx inside a subtree view to a global node_idx"""
        return node_idx + self.root_idx

    # functions below return modify and return a new tree
    @validate_index
    def delete_node(self, node_idx: int, replacement_token: Optional[Any] = None):
        """
        Returns a new tree with branch at node_idx deleted or replaced with a single node without children.

        Does the following, given this tree and node_idx=2, replacement_token=99:

        0 MethodDeclaration
        ├── 1 parameters
        │   ├── 2 FormalParameter
        │   │   ├── 3 type
        │   │   │   └── 4 ReferenceType
        │   │   │       └── 5 name
        │   │   │           └── 6 Bitmap
        │   │   └── 7 name
        │   │       └── 8 bmp
        │   └── 9 FormalParameter
        │       ├── 10 type
        │       │   └── 11 ReferenceType
        │       │       └── 12 name
        │       │           └── 13 File
        │       └── 14 name
        └── 15 body

        Will return tensors for the following tree:

        0 MethodDeclaration (0)
        ├── 1 parameters (1)
        │   ├── 2 <MASK> (2)
        │   └── 9 FormalParameter (3)
        │       ├── 10 type (4)
        │       │   └── 11 ReferenceType (5)
        │       │       └── 12 name (6)
        │       │           └── 13 File (7)
        │       └── 14 name (8)
        └── 15 body (9)

        For node_idx=2, replacement_token=None:

        0 MethodDeclaration (0)
        ├── 1 parameters (1)
        │   └── 9 FormalParameter (3)
        │       ├── 10 type (4)
        │       │   └── 11 ReferenceType (5)
        │       │       └── 12 name (6)
        │       │           └── 13 File (7)
        │       └── 14 name (8)
        └── 15 body (9)


        The original tensors have been:
        nodes:        [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        parents:       [ -1,  0,  1,  2,  3,  4,  5,  2,  7,  1,  9, 10, 11, 12,  9,  0]
        #descendants:  [ 14, 13,  6,  3,  2,  1,  0,  1,  0,  5,  3,  2,  1,  0,  0,  0]

        This method will return the following tensors:
        nodes:        [  0,  1, 99,  9, 10, 11, 12, 13, 14, 15]
        parents:      [ -1,  0,  1,  1,  3,  4,  5,  6,  3,  0]
        #descendants: [  9,  7,  0,  5,  3,  2,  1,  0,  0,  0]
        """
        return tensortree.delete_subtree(self, node_idx, replacement_token)

    def delete_nodes(
        self,
        node_indices: Union[int, torch.Tensor],
        replacements: Optional[Any] = None,
        replacement_mask: torch.BoolTensor = None,
        return_node_indices: bool = False,
        return_leaves: bool = False,
    ):
        """
        Returns a new tree with branches at node_indices deleted or replaced with a single node. This is way
         more efficient than calling `delete_node()` multiple times!

        Parameters:
        'replacement_mask' selects for which indices a replacement token should be inserted and
        must be of same length as `node_indices` (and `len(replacements) == sum(replacement_mask)`).
        With `return_node_indices=True` this method additionally returns the indices of the inserted
        in the new tree.
        With `return_leaves`=True this returns a single tensor of `node_data` with all leaf data. This
        requires less computation.

        Warning: The node indices must be in distinct branches, this is not checked!

        Does the following, given the following tree and
            - node_indices=[2, 12, 14],
            - replacements=['MASK1', 'MASK2']
            - replacement_mask=[True, False, True]

        0 MethodDeclaration
        ├── 1 parameters
        │   ├── 2 FormalParameter           # whole branch will be replaced with MASK1
        │   │   ├── 3 type
        │   │   │   └── 4 ReferenceType
        │   │   │       └── 5 name
        │   │   │           └── 6 Bitmap
        │   │   └── 7 name
        │   │       └── 8 bmp
        │   └── 9 FormalParameter
        │       ├── 10 type
        │       │   └── 11 ReferenceType
        │       │       └── 12 name         # whole branch will be deleted
        │       │           └── 13 File
        │       └── 14 name                 # whole branch will be replaced with MASK2
        └── 15 body

        Will return the following tree:

        0 MethodDeclaration (0)
        ├── 1 parameters (1)
        │   ├── 2 MASK1 (2)
        │   └── 9 FormalParameter (3)
        │       ├── 10 type
        │       │   └── 11 ReferenceType
        │       └── 14 MASK3 (8)
        └── 15 body (9)

        The original tensors have been:
        nodes:        [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        parents:       [ -1,  0,  1,  2,  3,  4,  5,  2,  7,  1,  9, 10, 11, 12,  9,  0]
        #descendants:  [ 14, 13,  6,  3,  2,  1,  0,  1,  0,  5,  3,  2,  1,  0,  0,  0]

        This method will return the following tensors:
        nodes:        [  0,  1, 99,  9, 10, 11, 12, 13, 14, 15]
        parents:      [ -1,  0,  1,  1,  3,  4,  5,  6,  3,  0]
        #descendants: [  9,  7,  0,  5,  3,  2,  1,  0,  0,  0]
        """

        if return_leaves:
            # then this method will return only the tensor with leaf tokens. speed up calculation.
            if replacement_mask is not None:
                raise NotImplementedError(
                    "Replacement_mask is not supported yet."
                )  # fixme

            return tensortree.delete_nodes_and_return_leaves(
                self, node_indices, replacements
            )

        return tensortree.delete_nodes(
            self, node_indices, replacements, replacement_mask, return_node_indices
        )  # returns a Tree

    def delete_siblings(
        self,
        node_indices: int,
        replacement_token: Optional[Any] = None,
        dont_check: bool = False,
    ):
        """
         Returns a new tree with branches at node_indices deleted or replaced with a single node without children.
        The node indices must be direct siblings!
        """
        if not dont_check:
            from itertools import zip_longest

            first_node = node_indices[0]
            sibs = self.iter_siblings(first_node, include_left_siblings=False)

            for sibling_idx, node_idx in zip_longest(
                sibs, node_indices[1:], fillvalue=None
            ):
                if node_idx is None:
                    break

                if node_idx != sibling_idx:
                    raise ValueError("Node indices seem to be no direct siblings.")

        return tensortree.delete_siblings(self, node_indices, replacement_token)

    @validate_index
    def delete_children(self, node_idx: int, replacement_token: Optional[Any] = None):
        return tensortree.delete_children(self, node_idx, replacement_token)

    def delete_children_of_nodes(
        self,
        node_indices: Union[int, torch.Tensor],
        replacements: Optional[Any] = None,
        replacement_additional_data: list[list[Any]] = None,
    ):
        """
        Similar to `tree.delete_nodes(...)` but replaces the children of the given `node_indices`.
        """

        return tensortree.replace_children_of_nodes(
            self, node_indices, replacements, replacement_additional_data
        )

    @validate_index
    def swap(self, node_idx: int, other_node_idx: int):
        return tensortree.swap(self, node_idx, other_node_idx)

    def insert_child(
        self, parent_idx: int, node_data: Any, right_sibling_idx: Optional[int] = None
    ):
        """adds a node (or a TensorTree) as a child of node at parent_idx, so that it is the left sibling of
        node at right_sibling_idx. If right_sibling_idx is None then it will be appended as the last child.
        """

        return tensortree.insert_child(self, parent_idx, node_data, right_sibling_idx)

    @staticmethod
    def concatenate(
        trees: list, new_root_node_data: Any, new_root_additional_data: list[Any] = None
    ):
        """
        Places trees under a new root to create one large tree.

        Does the following, given the following tree and
            - trees=[tree1, tree2]
            - new_root_node_data='ROOT'

        tree 1:
          0 MethodDeclaration
            └── 1 parameters
        tree 2:
          0 MethodDeclaration
            └── 1 parameters

        Returns the following tree:
        0 ROOT
            ├── 1 MethodDeclaration
            │   └── 2 parameters
            └── 3 MethodDeclaration
                └── 4 parameters
        """
        return tensortree.cat(trees, new_root_node_data, new_root_additional_data)

    def truncate(
        self, length: int, count_leaves: bool = False, removed_nodes_mask: bool = False
    ) -> Union["Tree", tuple["Tree", torch.BoolTensor]]:
        """
        Truncates this tree to contain at max `length` amount of nodes (or leaves).
        This method takes care that a nonterminal does not become a leaf after truncation.
        """
        return self.crop(
            length, count_leaves=count_leaves, removed_nodes_mask=removed_nodes_mask
        )

    def crop(
        self,
        length: int,
        start_idx: int = 0,
        count_leaves: bool = False,
        removed_nodes_mask: bool = False,
    ) -> Union["Tree", tuple["Tree", torch.BoolTensor]]:
        """
        Crops this tree to start with leaf at `start_idx` and end with leaf at `end_idx`.
        All nonterminals that will have no children will be removed.
        """
        nodes_to_remove = []

        leaf_mask = self.leaf_mask()
        nodes = leaf_mask.nonzero().view(-1) if count_leaves else range(len(self))

        if start_idx:
            if start_idx > len(self):
                raise ValueError(
                    f"Start index {start_idx} is out of bounds for tree with len {len(self)} ({len(nodes)} nodes to consider)."
                )
            elif count_leaves and not self.is_leaf(start_idx):
                raise ValueError("Start index is not a leaf.")
            elif not count_leaves:
                raise NotImplementedError("Implement this")
        else:
            if len(self) < length:
                return self

        if start_idx:
            # gather nodes to remove at the beginning
            current_node = start_idx  # last node to keep
            while current_node:
                branch_to_remove = self.left_sibling(current_node)
                if branch_to_remove is not None:
                    nodes_to_remove.append(branch_to_remove)
                    current_node = branch_to_remove
                else:
                    current_node = self.get_parent(current_node)

        # gather nodes to remove at the end
        num_leaves_before = leaf_mask[:start_idx].sum().item() if count_leaves else 0

        if num_leaves_before + length < len(nodes):
            current_node = nodes[num_leaves_before + length]  #  node to remove

            # walk upwards until we find a node that is not the first child of its parent
            while (
                current_node == self.get_parent(current_node) + 1
                and current_node is not None
            ):
                current_node = self.get_parent(current_node)

            if current_node is None:
                return  # would remove everything

            while current_node is not None and current_node < len(self):
                nodes_to_remove.append(current_node)
                current_node = self.next_node_not_in_branch(current_node)

        if not nodes_to_remove:
            return self

        nodes_to_remove = torch.tensor(sorted(nodes_to_remove))
        new_tree = self.delete_nodes(nodes_to_remove)

        if removed_nodes_mask:
            # create a mask of nodes that will be removed
            subtree_masks = self.subtree_masks()
            removed_nodes_mask = subtree_masks[nodes_to_remove].any(dim=0)
            return new_tree, removed_nodes_mask

        return new_tree


def equals(tree1: TensorTree, tree2: TensorTree):
    if type(tree1.node_data) is not type(tree2.node_data):
        return False

    if len(tree1) != len(tree2):
        return False

    if torch.any(tree1.descendants != tree2.descendants):
        return False

    if torch.any(tree1.parents != tree2.parents):
        return False

    if len(tree1.additional_data) != len(tree2.additional_data):
        return False

    if isinstance(tree1.node_data, list):
        if any(x1 != x2 for x1, x2 in zip(tree1.node_data, tree2.node_data)):
            return False

        for data1, data2 in zip(tree1.additional_data, tree2.additional_data):
            if any(x1 != x2 for x1, x2 in zip(data1, data2)):
                return False

    else:
        if torch.any(tree1.node_data != tree2.node_data):
            return False

        for data1, data2 in zip(tree1.additional_data, tree2.additional_data):
            if torch.any(data1 != data2):
                return False

    return True
