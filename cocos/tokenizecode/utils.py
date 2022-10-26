from typing import Union, Optional

from pathlib import Path

from cocos.tensortree import TensorTree


HAS_MAGIC = False
try:
    import magic

    HAS_MAGIC = True
except ImportError:
    import warnings

    warnings.warn(
        "Please install python-magic to determine file encoding (requires libmagic"
        "to be available on your system)."
    )


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def smart_read_file(filepath: Union[str, Path]) -> str:
    """
    Tries to determine the encoding of the file and open it accordingly.
    :param filepath:
    :return:
    """

    def _read_file(filepath: Path, encoding: Optional[str]) -> str:
        with filepath.open("rt", encoding=encoding) as f:
            return f.read()

    filepath = Path(filepath)

    if not HAS_MAGIC:
        return _read_file(filepath)

    # determine encoding
    m = magic.Magic(mime_encoding=True)
    encoding = m.from_file(str(filepath))

    try:
        content = _read_file(filepath, encoding)
    except (UnicodeDecodeError, LookupError):
        # we get an encoding that open() doesn't know: LookupError: unknown encoding: unknown-8bit
        # --> try another time with utf-8 (default encoding)
        # print(f"[LookupError] Could not parse file: {filepath} -> parsing with 'utf-8'")
        content = _read_file(filepath, encoding=None)

    return content


def is_tree_of_strings(tree: TensorTree) -> bool:
    return isinstance(tree.node_data[0], str)


TensorTreeWithStrings = TensorTree  # with only strings as nodes
TensorTreeWithInts = TensorTree  # with ints (ids) as nodes

from cocos.tokenizecode.parser import Span


def adjust_positions(
    tree1: TensorTreeWithStrings,
    tree2: TensorTreeWithInts,
    positions: list[Span],
    bpe_id: int,
) -> list[Span]:
    """Inaccurate the span of a splitted token is just the span of its original word. If"""
    assert len(positions) == len(tree1)

    result: list[Span] = []
    old_idx = 0
    for node_idx, node in enumerate(tree2.node_data):
        if node == bpe_id:
            # at added subword token
            span = positions[old_idx]
            old_idx += 1
        elif node_idx > 0 and tree2.get_node_data(tree2.get_parent(node_idx)) == bpe_id:
            # at subword for which we dont have a span
            span = None
        else:
            span = positions[old_idx]
            old_idx += 1

        result.append(span)

    assert len(result) == len(tree2)
    return result


# def adjust_positions_for_subwords(tree1: TensorTreeWithStrings, tree2: TensorTreeWithStrings, positions: list[Span], bpe_token: str) -> list[Span]:
#     """
#     Constructs exact positions, even for subwords.
#
#     :param tree1: The original parse tree (no bpe)
#     :param tree2: The bpe tree (decoded) with strings a tokens. (bpe)
#     :param positions: The original spans (no bpe)
#     :param bpe_token: bpe nonterminal
#     :return:
#     """
#     assert len(positions) == len(tree1)
#
#     result: list[Span] = []
#
#     parents_span: Span = None
#     offset = 0
#     row_offset = 0
#
#     old_idx = 0
#     for node_idx, node in enumerate(tree2.node_data):
#         if node == bpe_token:
#             # at added subword token
#             old_idx += 1
#             span = positions[old_idx]
#             parents_span = span
#         elif tree2.get_parent(node_idx) == bpe_token:
#             # at subword for which we dont have a span
#             node_length = len(node)
#             start_byte = parents_span.start_byte + offset
#             end_byte = start_byte + node_length
#
#             node_without_newlines = node.replace("\n", "")
#             rows = len(node) - len(node_without_newlines)
#
#             start_point = Point(
#                 row=parents_span.start_point.row + row_offset,
#                 column=parents_span.start_point.row,
#             )
#
#             end_point = Point(
#                 row=parents_span.start_point.row + row_offset,
#                 column=parents_span.start_point.row,
#             )
#
#             offset += node_length
#             row_offset += rows
#
#             span = Span(
#                 start_byte=start_byte,
#                 end_byte=end_byte,
#                 start_point=,
#                 end_point=parents_span.end_point
#             )
#         else:
#             old_idx += 1
#             span = positions[old_idx]
#
#         result.append(span)
#
#     assert len(result) == tree2
#     return result
