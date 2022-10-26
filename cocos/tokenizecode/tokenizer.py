from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import tokenizers
import torch
import transformers

from cocos.tensortree import TensorTree

from cocos.tokenizecode.bpe import TokenizerBPE
from cocos.tokenizecode.parser import CodeParser, CodeParsingOutput, Span
from cocos.tokenizecode.utils import (
    TensorTreeWithStrings,
    TensorTreeWithInts,
    is_tree_of_strings,
    get_project_root,
)


DEFAULT_TOKENIZER_BPE = get_project_root() / "tokenizer.json"


@dataclass
class TokenizedCodeOutput(CodeParsingOutput):
    tree: TensorTreeWithInts  # just changing the type

    def __post_init__(self):
        if len(self.positions) != len(self.tree):
            raise ValueError("Should have a position for every node in the tree.")


class CodeTokenizer:
    """Combines tokenizer and bpe."""

    def __init__(
        self,
        tokenizer: Optional[TokenizerBPE] = None,
        parser: Optional[CodeParser] = None,
    ):
        if tokenizer is None:
            tokenizer = TokenizerBPE.from_pretrained(DEFAULT_TOKENIZER_BPE)

        self.tokenizer = tokenizer
        self._parser = parser if parser is not None else None

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.hf_tokenizer.vocab == other.hf_tokenizer.vocab
        )

    @property
    def hf_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.tokenizer.tokenizer

    @property
    def parser(self):
        if self._parser is None:
            self._parser = CodeParser()

        return self._parser

    def _inputs_to_tree(
        self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None
    ) -> TensorTreeWithStrings:
        """Either parses a piece of code or uses the tree."""
        if isinstance(inputs, str):
            code = inputs

            if lang is None:
                raise ValueError("Will parse code. Language needs to be known.")

            parsing_output = self.parse(code, lang)
            return parsing_output.tree

        elif isinstance(inputs, CodeParsingOutput):
            tree = inputs.tree
        else:
            tree = inputs

        if not is_tree_of_strings(tree):
            raise ValueError("Tree should consist of strings.")

        return tree

    @classmethod
    def from_file(cls, tokenizer_file_or_directory: Path):
        from .bpe import TokenizerBPE

        return cls(TokenizerBPE.from_pretrained(tokenizer_file_or_directory))

    def parse(self, code: str, lang: str) -> CodeParsingOutput:
        """Turns a piece code into a syntax tree."""
        return self.parser.parse(code, lang)

    @staticmethod
    def unparse(tree: Union[TensorTreeWithStrings, CodeParsingOutput]) -> str:
        """Turns a syntax_tree tree back into code."""
        if isinstance(tree, CodeParsingOutput):
            tree = tree.tree

        if not is_tree_of_strings(tree):
            raise ValueError("Tree should consist of strings.")

        return CodeParser.unparse(tree)

    def encode(
        self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None
    ) -> tokenizers.Encoding:
        """Encodes a piece of code or a syntax tree and returns an encoding for all tokens."""
        tree = self._inputs_to_tree(inputs, lang)
        return self.tokenizer.encode_text(tree.leaves())

    def encode_text(self, text: Union[str, list[str]]) -> tokenizers.Encoding:
        """Encodes any piece of text or list of text."""
        return self.tokenizer.encode_text(text)

    def encode_to_tree(
        self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None
    ) -> TensorTreeWithInts:
        """Encodes a piece of code or a syntax tree and returns **the full syntax tree** encoded (ie only ids as nodes)."""

        tree = self._inputs_to_tree(inputs, lang)
        return self.tokenizer.encode_tree(tree)

    def encode_lines(
        self,
        code: str,
        lang: str,
        line_start: int,
        line_end: int,
        mask_line_start: int = None,
        mask_line_end: int = None,
    ) -> Union[tokenizers.Encoding, tuple[tokenizers.Encoding, tokenizers.Encoding]]:
        """
        Parses the whole file and then selects relevant lines. Lines start at 1. If mask line start is set, those lines
        will be cut out and the mask token is inserted.
        """
        line_start -= 1
        line_end -= 1
        mask_line_start = mask_line_start - 1 if mask_line_start is not None else None
        mask_line_end = mask_line_end - 1 if mask_line_end is not None else None
        assert line_start <= mask_line_start <= mask_line_end <= line_end

        output = self.parse(code, lang)
        tree = output.tree

        context_tokens = []
        mask_tokens = []

        for idx, (node, span) in enumerate(zip(tree.node_data, output.positions)):
            if not tree.is_leaf(idx):
                continue

            if line_start <= span.start_point.row <= line_end:
                if (
                    mask_line_start is not None
                    and mask_line_start == span.start_point.row
                ):
                    # keep first space
                    if node.isspace() and not mask_tokens:
                        context_tokens.append(node)
                        context_tokens.append("___MASK___")

                    mask_tokens.append(node)

                elif (
                    mask_line_start is not None
                    and mask_line_start <= span.start_point.row <= mask_line_end
                ):
                    mask_tokens.append(node)
                else:
                    context_tokens.append(node)

        if mask_line_start is not None:
            return self.tokenizer.encode_text(
                context_tokens
            ), self.tokenizer.encode_text(mask_tokens)

        return self.tokenizer.encode_text(context_tokens)

    def decode(self, ids) -> str:
        if isinstance(ids, TensorTreeWithInts):
            tree = ids
            decoded_tree = self.decode_tree(tree, keep_bpe=False)
            return self.unparse(decoded_tree)

        return self.tokenizer.decode_text(ids)

    def decode_tree(
        self, tree: TensorTreeWithInts, keep_bpe: bool = False
    ) -> TensorTreeWithStrings:
        """Returns a tree with strings as node data. keep_bpe keeps artificial BPE nodes."""
        return self.tokenizer.decode_tree(tree, keep_bpe)

    @staticmethod
    def tree_to_tokens(tree: TensorTree) -> Union[torch.Tensor, list[str]]:
        return tree.leaves()

    def add_specials(self, tokens):
        self.tokenizer.tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}
        )

    def save(self, filepath: Union[str, Path], pretty: bool = False):
        self.tokenizer.tokenizer.save_pretrained(str(filepath), legacy_format=False)
        # self.tokenizer.tokenizer.save(str(filepath), pretty)

    def __len__(self):
        return len(self.hf_tokenizer)
