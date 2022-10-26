from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import time
from typing import Union, Optional, Iterable
import warnings

import tree_sitter
from tree_sitter import Language, Parser

from cocos import tensortree
from cocos.tensortree import TensorTree
from cocos.tokenizecode.utils import TensorTreeWithStrings, get_project_root
from cocos.tokenizecode.point import Span, Point


log = logging.getLogger(__name__)


def download_grammar(
    language: str,
    directory: Path,
    sha: Optional[str] = None,
    release: Optional[str] = None,
):
    # loads specific commit if sha is given, else current master branch
    import requests, zipfile, io

    log.info(f"Cloning grammar for language {language} to {directory}.")
    if sha is not None:
        url = f"https://github.com/tree-sitter/tree-sitter-{language}/archive/{sha}.zip"
        path = directory / f"tree-sitter-{language}-{sha}"
        new_path = directory / f"tree-sitter-{language}/{sha}"
    elif release is not None:
        url = f"https://github.com/tree-sitter/tree-sitter-{language}/archive/refs/tags/{release}.zip"
        dir_name = release[1:] if release.startswith("v") else release
        path = directory / f"tree-sitter-{language}-{dir_name}"
        new_path = directory / f"tree-sitter-{language}/{release}"
    else:
        url = f"https://github.com/tree-sitter/tree-sitter-{language}/archive/refs/heads/master.zip"
        path = directory / f"tree-sitter-{language}-master"
        new_path = directory / f"tree-sitter-{language}/master"

    print(f"Downloading {url} to {path}")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(directory.absolute())

    shutil.move(
        str(path.absolute()),
        str(new_path.absolute()),
    )
    assert new_path.exists(), "this directory should exist"

    return new_path


class TreeSitterParser:
    """A treesitter code parser that magically downloads and initializes itself"""

    # these are the versions the tokenizer was trained with
    _versions = {
        "c": "v0.20.0",
        "cpp": "v0.20.0",
        "css": "v0.20.0",
        "c-sharp": "v0.20.0",
        "haskell": "v0.13.0",
        "html": "v0.20.0",
        "go": "v0.20.0",
        "java": "v0.20.0",
        "javascript": "v0.19.0",
        "json": "v0.19.0",
        "julia": "v0.20.0",
        "ocaml": "v0.20.0",
        "php": "v0.20.0",
        "python": "v0.20.0",
        "ruby": "v0.19.0",
        "rust": "v0.20.0",
        "scala": "v0.20.0",
        "typescript": "v0.19.0",
        # 'agda', 'swift', 'verilog' currently bugged. see: https://github.com/tree-sitter/py-tree-sitter/issues/72
    }
    supported_languages = set(_versions.keys())

    def __init__(
        self,
        libs_dir: Optional[Path] = None,
        grammar_versions: Optional[dict] = None,
        load_newest_grammars: bool = False,
    ):
        if libs_dir is None:
            self.libs_dir = get_project_root() / "cocos/tokenizecode/libs"

        # we use the default versions and overwrite them with the given ones
        versions_to_use = self._versions.copy() if not load_newest_grammars else {}
        self.grammar_versions = (
            {**versions_to_use, **grammar_versions}
            if grammar_versions is not None
            else versions_to_use
        )
        # initialize the missing languages with None
        self.grammar_versions = {
            l: self.grammar_versions[l] if l in self.grammar_versions else None
            for l in self.supported_languages
        }

        self.build_path = (self.libs_dir / "langs.so").absolute()

        self.language = None
        self.LANGUAGES: dict[str, Language] = {}
        self._setup_grammars()
        self.parser = Parser()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["parser"]
        del state["LANGUAGES"]
        del state["language"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

        self.language = None
        self.LANGUAGES: dict[str, Language] = {}
        self._setup_grammars()
        self.parser = Parser()

    def _setup_grammars(self):
        # build already. call reset
        if self.LANGUAGES:
            return

        self.libs_dir.mkdir(parents=True, exist_ok=True)

        downloaded_langs = {}

        for d in filter(
            lambda d: d.is_dir() and d.name.startswith("tree-sitter-"),
            self.libs_dir.iterdir(),
        ):
            l = d.name.replace("tree-sitter-", "")
            version = self.grammar_versions[l] or "master"
            if version in (subdir.name for subdir in d.iterdir()):
                downloaded_langs[l] = d / version

        did_download = False

        for language in (
            l for l in self.supported_languages if l not in downloaded_langs
        ):
            downloaded_langs[language] = download_grammar(
                language,
                self.libs_dir,
                release=self.grammar_versions[language],
            )
            did_download = True

        # wait a little after downloading
        if did_download:
            self.build_path.unlink(missing_ok=True)
            time.sleep(1)

        build_path = str(self.build_path.absolute())
        language_directories = []

        for lang, dir_ in downloaded_langs.items():
            src_dir = dir_ / "src"
            if src_dir.exists():
                language_directories.append(str(dir_.absolute()))
                continue

            src_dir = dir_ / lang / "src"
            if src_dir.exists():
                language_directories.append(str(src_dir.parent.absolute()))
                continue

            src_dir = dir_ / "grammars" / lang / "src"
            if src_dir.exists():
                language_directories.append(str(src_dir.parent.absolute()))
                continue
            else:
                warnings.warn(f"No 'src' folder found for language under {dir_}")
                self.supported_languages.remove(lang)

        did_build = Language.build_library(build_path, language_directories)

        if did_build:
            time.sleep(1)

        self.LANGUAGES = {}
        for lang in downloaded_langs:
            try:
                ts_language = Language(build_path, lang.replace("-", "_"))
                self.LANGUAGES[lang] = ts_language
            except AttributeError as e:
                print(e)

    def _set_language(self, language: str):
        if language not in self.supported_languages:
            raise ValueError(f"No parser for language {language}")

        self.parser.set_language(self.LANGUAGES[language])
        self.language = language

    def reset(self, reload_all: bool = False):
        if reload_all and self.libs_dir.exists():
            import shutil

            shutil.rmtree(self.libs_dir)

        self.LANGUAGES = {}
        self._setup_grammars()

    def parse(self, code: Union[str, bytes], language: str = None):
        if self.language is None and language is None:
            raise ValueError("provide language")

        if language != self.language:
            self._set_language(language)

        if isinstance(code, str):
            code = code.encode("utf-8")

        return self.parser.parse(code)


@dataclass
class _TmpNode:
    """Helper during parsing."""

    id_: int
    text: str
    parent_id: int
    descendants: int

    span: Span

    def __post_init__(self):
        if isinstance(self.text, bytes):
            self.text = self.text.decode("utf-8")


class TreeTraversal:
    """Implement a call method, that takes code and a treesitter and produces an iterable of nodes."""

    def __init__(self):
        self.errors = 0  # number of errors in last traversal

    def __call__(self, code: str, tree: tree_sitter.Tree) -> Iterable[_TmpNode]:
        raise NotImplementedError


class FullTraversal(TreeTraversal):

    def __call__(self, code: str, tree: tree_sitter.Tree) -> Iterable[_TmpNode]:
        self.errors = 0  # will be incremented in traverse

        nodes = list(self.traverse_tree(code, tree))

        # if self.errors:
        #     import warnings
        #     warnings.warn(f"Found {self.errors} errors while parsing. Is the parser set to the correct language?")

        return nodes

    def traverse_tree(self, code: str, tree: tree_sitter.Tree):
        """Fuck, this code is a mess... but it magically works."""

        if isinstance(code, str):
            code = code.encode("utf-8")

        cursor = tree.walk()

        reached_root = False

        last_end_byte: int = None
        last_end_point: Point = None

        node_id = -1
        open_nodes: list[_TmpNode] = []
        root: Optional[_TmpNode] = None

        def text_between(node_start_byte, node_start_point):
            # nonlocal last_leaf_checked, last_leaf
            nonlocal last_end_byte, last_end_point

            if last_end_byte is None:
                return

            if last_end_byte < node_start_byte:
                text = code[last_end_byte:node_start_byte]
                _last_end_point = last_end_point
                _last_end_byte = last_end_byte

                last_end_byte = node_start_byte
                last_end_point = node_start_point
                return text, _last_end_point, _last_end_byte

        def add_node(text, span):
            nonlocal node_id, root

            node_id += 1
            if open_nodes:
                parent_id = open_nodes[-1].id_
                open_nodes[-1].descendants += 1
            elif last_end_byte:
                # at the end
                parent_id = 0  # root
            else:
                parent_id = -1

            if text == "[ERROR]":
                self.errors += 1

            node = _TmpNode(
                id_=node_id, text=text, parent_id=parent_id, descendants=0, span=span
            )

            if parent_id == -1:
                root = node

            return node

        def to_node(node, read_text: bool):
            nonlocal last_end_byte, last_end_point

            span = Span(
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_point=Point(*node.start_point),
                end_point=Point(*node.end_point),
            )
            text = (
                code[span.start_byte : span.end_byte] if read_text else f"[{node.type}]"
            )
            node = add_node(text, span)

            if read_text:
                last_end_byte = node.span.end_byte
                last_end_point = node.span.end_point

            return node

        while not reached_root:
            code_between = text_between(cursor.node.start_byte, cursor.node.start_point)
            if code_between:
                text, _last_end_point, _last_end_byte = code_between
                code_span = Span(
                    start_byte=_last_end_byte,
                    end_byte=cursor.node.start_byte,
                    start_point=_last_end_point,
                    end_point=Point(*cursor.node.start_point),
                )
                yield add_node(text, code_span)

            if cursor.node.is_named and not cursor.node.children:
                node = to_node(cursor.node, read_text=False)
                open_nodes.append(node)
                yield node

            node = to_node(cursor.node, read_text=not bool(cursor.node.children))
            yield node

            if cursor.node.is_named and not cursor.node.children:
                if len(open_nodes) > 1:
                    open_nodes[-2].descendants += open_nodes[-1].descendants
                open_nodes = open_nodes[:-1]

            if cursor.goto_first_child():
                open_nodes.append(node)
                continue

            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                else:
                    if len(open_nodes) > 1:
                        open_nodes[-2].descendants += open_nodes[-1].descendants

                    open_nodes = open_nodes[:-1]

                if cursor.goto_next_sibling():
                    retracing = False

        code_between = text_between(cursor.node.end_byte, cursor.node.end_point)
        if code_between:
            text, _last_end_point, _last_end_byte = code_between
            code_between_span = Span(
                start_byte=_last_end_byte,
                end_byte=cursor.node.end_byte,
                start_point=_last_end_point,
                end_point=Point(*cursor.node.end_point),
            )
            yield add_node(text, code_between_span)
            root.descendants += 1


def to_tensortree(nodes: list[_TmpNode]) -> tuple[TensorTree, list[Span]]:
    # correct descendants value will be set after iteration (!)
    node_data, parents, descendants, positions = [], [], [], []

    for node in nodes:
        # node_data.append(replace_whitespace(node.text))
        node_data.append(node.text)
        parents.append(node.parent_id)
        descendants.append(node.descendants)
        positions.append(node.span)

    return tensortree.tree(parents, node_data, descendants), positions


@dataclass
class CodeParsingOutput:
    tree: TensorTreeWithStrings
    positions: list[Span]
    num_errors: int
    language: str

    def __post_init__(self):
        if len(self.positions) != len(self.tree):
            raise ValueError("Should have a position for every node in the tree.")


class CodeParser:
    """
    Computes the full syntax tree with all tokens for a piece of code.
    """

    def __init__(self, traversal: Optional[TreeTraversal] = None):
        self.parser = TreeSitterParser()
        self.traverse = traversal if traversal is not None else FullTraversal()

    @staticmethod
    def supported_languages() -> set[str]:
        return TreeSitterParser.supported_languages

    def parse(self, code: str = None, lang: str = "java") -> CodeParsingOutput:
        ts_tree = self.parser.parse(code, lang)
        nodes = list(self.traverse(code, ts_tree))
        num_errors = self.traverse.errors
        tree, positions = to_tensortree(nodes)

        return CodeParsingOutput(tree, positions, num_errors=num_errors, language=lang)

    @staticmethod
    def unparse(tree: TensorTree) -> str:
        if isinstance(tree, CodeParsingOutput):
            tree = tree.tree
        return "".join(tree.leaves())

    def pprint(self, tree: TensorTree) -> None:
        print(self.unparse(tree))
