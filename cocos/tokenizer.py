from functools import lru_cache
import logging
import re
import torch
from typing import Union

from cocos.tokenizecode import CodeTokenizer
from cocos import tensortree

from cocos.source_code import ALL_IDENTIFIER_NODE, LANGUAGES
from cocos.utils import get_project_root


log = logging.getLogger(__name__)


# separates those specials from other tokens in decoded text
RE_SPECIALS = re.compile(r"(VAR\d+|FOLD\d+)")
RE_MASK = re.compile(r"___MASK(\d*)___")
RE_SPECIALS = re.compile(
    r"(VAR\d+|FOLD\d+|___MASK(\d*)___|\[SEP]|\[PAD]"
    r"\[C]|\[C-SHARP]|\[CPP]|\[CSS]|\[GO]|\[HASKELL]|\[JAVA]|\[JAVASCRIPT]|\[JULIA]|\[OCAML]"
    r"|\[PHP]|\[PYTHON]|\[RUBY]|\[RUST]|\[SCALA]|\[TYPESCRIPT])"
)


LANGUAGE_TOKEN_MAP = {language: f"[{language.upper()}]" for language in LANGUAGES}

LANGUAGE_TOKEN_TO_LANG = {v: k for k, v in LANGUAGE_TOKEN_MAP.items()}


METHOD_NONTERMINALS: set[str] = {
    "[function_declaration]",
    "[method_declaration]",
    "[function_declarator]",
    "[method_definition]",
    "[method]",
    "[function_definition]",
    "[function]",
    "[class_function]",
}


TOKENIZER: CodeTokenizer = None


####################################################################
# strings
@lru_cache(maxsize=1)
def fold_tokens(n: int = 1):
    return [f"FOLD{i}" for i in range(n)]


@lru_cache(maxsize=1)
def language_tokens() -> list[str]:
    return sorted(list(LANGUAGE_TOKEN_MAP.values()))


@lru_cache(maxsize=1)
def mask_tokens(n: int = 500):
    return [f"___MASK{i}___" for i in range(n)]


@lru_cache(maxsize=1)
def method_nonterminals() -> list[str]:
    return sorted(list(METHOD_NONTERMINALS))


# ids
@lru_cache(maxsize=1)
def get_fold_token_ids(n: int = 1) -> torch.Tensor:
    tokenizer = load_tokenizer()
    folding_replacement_tokens = tokenizer.hf_tokenizer.convert_tokens_to_ids(
        fold_tokens(n)
    )
    return torch.tensor(folding_replacement_tokens, dtype=torch.long)


@lru_cache(maxsize=1)
def get_language_token_ids() -> torch.Tensor:
    tokenizer = load_tokenizer()
    tokens = tokenizer.hf_tokenizer.convert_tokens_to_ids(language_tokens())
    return torch.tensor(tokens, dtype=torch.long)


@lru_cache(maxsize=1)
def get_mask_token_ids() -> torch.Tensor:
    tokenizer = load_tokenizer()

    tokens = [get_mask_id()]
    tokens += tokenizer.hf_tokenizer.convert_tokens_to_ids(mask_tokens())
    return torch.tensor(tokens, dtype=torch.long)


@lru_cache(maxsize=1)
def get_identifier_nonterminal():
    tokenizer = load_tokenizer()
    identifier_nonterminal = tokenizer.hf_tokenizer.convert_tokens_to_ids(
        ALL_IDENTIFIER_NODE
    )
    return identifier_nonterminal


@lru_cache(maxsize=1)
def get_mask_id():
    tokenizer = load_tokenizer()
    mask_token = tokenizer.hf_tokenizer.convert_tokens_to_ids("___MASK___")
    return mask_token


@lru_cache(maxsize=1)
def get_cls_id():
    tokenizer = load_tokenizer()
    token = tokenizer.hf_tokenizer.convert_tokens_to_ids("[CLS]")
    return token


@lru_cache(maxsize=1)
def get_pad_id():
    tokenizer = load_tokenizer()
    pad_token = tokenizer.hf_tokenizer.convert_tokens_to_ids("[PAD]")
    return pad_token


@lru_cache(maxsize=1)
def get_eos_id():
    tokenizer = load_tokenizer()
    pad_token = tokenizer.hf_tokenizer.convert_tokens_to_ids("[SEP]")
    return pad_token


@lru_cache(maxsize=1)
def get_sep_id():
    tokenizer = load_tokenizer()
    sep_token = tokenizer.hf_tokenizer.convert_tokens_to_ids("[SEP]")
    return sep_token


@lru_cache(maxsize=1)
def get_method_nonterminal_ids() -> torch.Tensor:
    tokenizer = load_tokenizer()

    tokens = tokenizer.hf_tokenizer.convert_tokens_to_ids(method_nonterminals())

    return torch.tensor(tokens, dtype=torch.long)


################################################################


def get_language_token(language: str) -> str:
    return LANGUAGE_TOKEN_MAP[language]


@lru_cache(maxsize=20)
def get_language_token_id(language: str) -> int:
    tokenizer = load_tokenizer()
    return tokenizer.hf_tokenizer.convert_tokens_to_ids(LANGUAGE_TOKEN_MAP[language])


def get_language(language_token: str) -> str:
    return LANGUAGE_TOKEN_TO_LANG[language_token]


def format_specials(target_str: str) -> str:
    s = RE_SPECIALS.sub(r" \1 ", target_str)
    s = RE_MASK.sub(r" [MASK\1] ", s)
    return s


def remove_specials(target_str: str) -> str:
    s = RE_SPECIALS.sub("", target_str)
    return s


def format_masks(code: str, surround_spaces: bool = True) -> str:
    return RE_MASK.sub(r" [MASK\1] " if surround_spaces else r"[MASK\1]", code)


def create_tokenizer() -> CodeTokenizer:
    tok = CodeTokenizer.from_file(get_project_root() / "tokenizer.json")
    tok.add_specials([ALL_IDENTIFIER_NODE])
    tok.add_specials(fold_tokens())
    tok.add_specials(language_tokens())
    tok.add_specials(mask_tokens())

    return tok


def load_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        log.info("Loading tokenizer from file.")
        TOKENIZER = (
            create_tokenizer()
        )  # fixme replace with saved tokenizer once everything is setup
    return TOKENIZER


#####################################
#####################################
# very tokenizer specific stuff

NUM_SPACES_IN_TOKEN = (
    [0] * 11  # special characters
    + [8, 4, 2, 1]  # space tokens
    + [0] * 40000  # rest of vocabulary
)
NUM_SPACES_IN_TOKEN = torch.tensor(NUM_SPACES_IN_TOKEN)


def is_newline(idx):
    if idx == 10:
        return True
    elif idx == 9:
        return True

    return False


def num_newline_in_token(token_id):
    if token_id == 10:
        return 2
    elif token_id == 9:
        return 1

    return 0


def is_whitespace(tokens: Union[int, torch.Tensor]) -> bool:
    if isinstance(tokens, torch.Tensor):
        return torch.all(tokens <= 14).bool()

    return tokens <= 14


def is_whitespace_node(tree: tensortree.TensorTree, node_idx: int) -> bool:
    # fixme this depends on a specific tokenizer. Whitespace needs to have an id below 15
    return is_whitespace(tree[node_idx].node_data)
