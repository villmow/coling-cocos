from hashlib import blake2b
import logging
from pathlib import Path
from typing import Union, Optional

import datasets
from omegaconf import OmegaConf, DictConfig
from cocos import tensortree


log = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    log.info(f"Loading following config from {config_path}:")
    log.info(OmegaConf.to_yaml(config))
    return config


def to_tree(
    sample: dict,
    key_prefix: str = "",
    additional_data_fields: Optional[Union[str, list[str]]] = "hashes",
) -> Union[tensortree.TensorTree, list[tensortree.TensorTree]]:
    """assumes trees are saved as key_prefix + "tokens", "descendants", "parents" """
    batched = False
    try:
        len(sample[key_prefix + "parents"][0])  # fails if this is not iterable
        batched = True
    except TypeError:
        pass

    if isinstance(additional_data_fields, str):
        additional_data_fields = [additional_data_fields]
    additional_data_fields = [f for f in additional_data_fields if f in sample]

    if batched:
        # batched, then everything else is a list too
        bsz = len(sample[key_prefix + "parents"])
        return [
            tensortree.tree(
                parents=sample[key_prefix + "parents"][i],
                descendants=sample[key_prefix + "descendants"][i],
                node_data=sample[key_prefix + "tokens"][i],
                additional_data=[sample[f][i] for f in additional_data_fields],
            )
            for i in range(bsz)
        ]
    else:
        return tensortree.tree(
            parents=sample[key_prefix + "parents"],
            descendants=sample[key_prefix + "descendants"],
            node_data=sample[key_prefix + "tokens"],
            additional_data=[sample[f] for f in additional_data_fields],
        )


def shorten_code_dataset(dataset: datasets.Dataset, num_files_per_language: int):
    """Selects first num_files_per_language samples for each language. Consider shuffling dataset beforehand."""
    log.info(
        f"Selecting up to {num_files_per_language} samples per language from {len(dataset)} samples."
    )

    # count
    log.info(f"Counting languages:")
    from collections import Counter
    from pprint import pformat

    c = Counter(dataset["language"])
    log.info(pformat(str(c)))

    files_for_lang = {lang: 0 for lang in c}
    rows_to_keep = []
    incomplete_langs = set(files_for_lang.keys())

    log.info("Selecting rows from dataset:")
    for i, language in enumerate(dataset["language"]):
        if language in incomplete_langs:
            if files_for_lang[language] < num_files_per_language:
                files_for_lang[language] += 1
                rows_to_keep.append(i)
            else:
                incomplete_langs.remove(language)

        if not incomplete_langs:
            break

    log.info(pformat(str(files_for_lang)))
    return dataset.select(rows_to_keep)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def int_to_bytes(x: int) -> bytes:
    """Convert an integer to a byte."""
    return x.to_bytes((x.bit_length() + 7) // 8, "big")


def hash_subtrees(tree: tensortree.TensorTree) -> list[int]:
    """Computes a hash for every token in the tree. Equal subtrees have equal hashes."""
    hashes = [None] * len(tree)

    def get_hash(node_idx: int = 0):
        token = tree.get_node_data(node_idx)  # may be a string

        if isinstance(token, str):
            token_byte = token.encode("utf-8")
        else:
            token_byte = int_to_bytes(token.item())

        if tree.is_leaf(node_idx):
            hash_ = blake2b(digest_size=6, usedforsecurity=False)
            hash_.update(token_byte)
            hashes[node_idx] = int(hash_.hexdigest(), base=16)
            return 0, hash_
        else:
            height = 0
            hash_ = None

            for child_idx in tree.iter_children(node_idx):
                child_height, child_hash_ = get_hash(child_idx)
                curr_height = child_height + 1

                if hash_ is None:
                    hash_ = child_hash_
                    hash_.update(token_byte)
                else:
                    hash_.update(child_hash_.digest())

                if curr_height > height:
                    height = curr_height

            level_byte = int_to_bytes(height)
            hash_.update(level_byte)
            hashes[node_idx] = int(hash_.hexdigest(), base=16)
            return height, hash_

    get_hash()
    assert len(hashes) == len(tree)
    assert None not in hashes

    return hashes


def print_code(code: str, lang: str, pretty_print: bool = True):
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    if pretty_print:
        if lang == "php":
            code = f"<? php\n{code}"

        syntax = Syntax(
            code, language_for_pygments(lang), theme="monokai", line_numbers=True
        )
        console.print(syntax)
    else:
        print(code)


def language_for_pygments(lang: str):
    if lang == "cpp":
        return "c++"
    elif lang == "c-sharp":
        return "c#"
    else:
        return lang


import numpy as np


def find_runs(x: np.ndarray):
    """
    Find runs of consecutive items in an array.

    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def get_wandb_experiment(experiment_directory: str) -> str:
    run_dir = Path(experiment_directory)

    wandb_run_id = None
    wandb_dir = run_dir / "wandb"
    if wandb_dir.exists():
        # filname is "run-IDIDIDID.wandb
        run_ids = {p.name[4:-6] for p in wandb_dir.glob("latest-run/run-*.wandb")}
        if len(run_ids) > 1:
            raise ValueError("Found multiple wandb run_ids. Can't resume experiment")
        elif run_ids:
            wandb_run_id = run_ids.pop()
            return wandb_run_id

    raise ValueError("Could not find wandb experiment")


import pytorch_lightning as pl


def load_model(
    checkpoint_path,
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[DictConfig] = None,
    config_overrides: Optional[dict] = None,
) -> pl.LightningModule:
    from hydra.utils import to_absolute_path, get_class, instantiate
    from omegaconf import OmegaConf

    if config is None:
        from cocos.utils import load_config

        config = load_config(to_absolute_path(str(config_path)))

    if config_overrides is not None:
        if isinstance(config_overrides, DictConfig):
            config = OmegaConf.merge(config, config_overrides)
        else:
            for key, value in config_overrides.items():
                OmegaConf.update(config, key, value)

    model_target = config.model._target_
    model_cls = get_class(model_target)
    log.info(f"Loading model of type {model_cls}")

    # only instantiate all kwargs of this model, but not the model itself
    del config.model._target_
    kwargs = instantiate(config.model)
    config.model._target_ = model_target

    # pass all instantiated kwargs to load from checkpoint
    model = model_cls.load_from_checkpoint(
        to_absolute_path(str(checkpoint_path)), **kwargs
    )

    return model


def convert_checkpoint_to_huggingface(
    checkpoint_path: Path,
    config_path: Path,
    save_directory: Path,
    key_model: bool = False,
    query_model: bool = False,
    config_overrides: Union[dict, DictConfig] = None,
):
    from cocos.model import (
        DualRetriever,
        EncoderForRetrieval,
        EncoderForSequenceClassification,
    )

    model = load_model(
        checkpoint_path, config_path=config_path, config_overrides=config_overrides
    )
    model_to_save = None

    if isinstance(model, DualRetriever):
        if model.key_model == model.query_model:
            model_to_save = model.key_model
        else:
            # two different models
            if not key_model and not query_model:
                raise ValueError(
                    "Model has different encoders for query/key. Explicitly pass which model to save."
                )
            elif key_model:
                model_to_save = model.key_model
            elif query_model:
                model_to_save = model.query_model

        log.warning(
            f"Saving {model.__class__.__name__} without Projection/Classification layers."
        )

    elif isinstance(model, EncoderForRetrieval):
        model_to_save = model.model
    elif isinstance(model, EncoderForSequenceClassification):
        model_to_save = model.model
        log.warning(
            f"Saving {model.__class__.__name__} without Projection/Classification layers."
        )
    else:
        model_to_save = getattr(model, "model", None)

    if model_to_save is not None:
        model_to_save.save_pretrained(save_directory)
        log.info(
            f"Saved {model_to_save.__class__.__name__} model under {save_directory}."
        )
    else:
        log.warning("Could not save model.")


def cli_convert_checkpoint_to_huggingface():
    def argument_parser():
        import argparse

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Converts a LightningModule to a T5EncoderModel.",
        )
        parser.add_argument("checkpoint", help="Path to checkpoint file.", type=Path)
        parser.add_argument("config", help="Path to config file.", type=Path)
        parser.add_argument(
            "output", help="Model save directory. Must not exist.", type=Path
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "-q",
            "--query-model",
            help="Save query model (context) if model has separate encoders.",
            action="store_true",
        )
        group.add_argument(
            "-k",
            "--key-model",
            help="Save key model (passage) if model has separate encoders.",
            action="store_true",
        )

        parser.add_argument(
            "config_overrides",
            help="Overrrides for config values. Same notation as in OmegaConf.",
            type=str,
            nargs="*",
        )

        return parser

    parser = argument_parser()
    args = parser.parse_args()

    config_overrides = OmegaConf.from_dotlist(args.config_overrides)
    print(config_overrides)

    convert_checkpoint_to_huggingface(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        save_directory=Path(args.output),
        key_model=args.key_model,
        query_model=args.query_model,
        config_overrides=config_overrides,
    )
