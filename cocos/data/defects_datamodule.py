import logging
from pathlib import Path
from typing import List, Optional

import datasets
from datasets import load_dataset
from hydra.utils import to_absolute_path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cocos.tensortree.collate import collate_tokens
from cocos.tokenizer import (
    load_tokenizer,
    get_language_token_id,
    get_eos_id,
    get_pad_id,
    CodeTokenizer,
)

log = logging.getLogger(__name__)


def encode_sample(
    sample, tokenizer: CodeTokenizer, c_lang_id_token: int, eos_id: int, max_length=None
):
    code = sample["func"]

    # note 5420 files will produce parsing errors. we ignore them.
    out = tokenizer.parse(code, "c")
    encoding = tokenizer.encode(out)

    tokens = encoding.ids
    if max_length is not None:
        tokens = tokens[: max_length - 2]

    tokens = [c_lang_id_token] + tokens + [eos_id]
    return {"tokens": tokens}


def load_and_preprocess_dataset(
    data_dir, max_length=None, num_proc=1
) -> datasets.DatasetDict:
    dataset_dir = Path(to_absolute_path(str(data_dir)))
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(dataset_dir / "train.jsonl"),
            "validation": str(dataset_dir / "valid.jsonl"),
            "test": str(dataset_dir / "test.jsonl"),
        },
    )
    # do that here once, next time in setup will be cached
    ds = dataset.map(
        encode_sample,
        batched=False,
        num_proc=num_proc,
        fn_kwargs={
            "tokenizer": load_tokenizer(),
            "c_lang_id_token": get_language_token_id("c"),
            "eos_id": get_eos_id(),
            "max_length": max_length,
        },
    )
    return ds


def download_dataset(data_dir: Path, keep_intermediate_files: bool = True):
    """Downloads and extracts the dataset into data_dir."""
    import gdown
    import requests

    data_dir = Path(to_absolute_path(str(data_dir)))
    data_dir.mkdir(parents=True, exist_ok=True)

    # download hosted functions file
    function_filepath = data_dir / "function.json"
    if not function_filepath.exists():
        functions_url = (
            "https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF"
        )
        gdown.download(functions_url, str(function_filepath), quiet=False)

    # download mapping which function is in which split
    train_ids_url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Defect-detection/dataset/train.txt"
    valid_ids_url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Defect-detection/dataset/valid.txt"
    test_ids_url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Defect-detection/dataset/test.txt"

    def download(url):
        filename = url.rsplit("/")[-1]
        save_filepath = data_dir / filename
        if not save_filepath.exists():
            r = requests.get(url, allow_redirects=True)
            with save_filepath.open("wb") as f:
                f.write(r.content)
        return save_filepath

    train_ids_filepath, valid_ids_filepath, test_ids_filepath = list(
        map(download, (train_ids_url, valid_ids_url, test_ids_url))
    )
    preprocess(
        data_dir,
        function_filepath,
        train_ids_filepath,
        valid_ids_filepath,
        test_ids_filepath,
    )

    if not keep_intermediate_files:
        # cleanup files
        function_filepath.unlink()
        train_ids_filepath.unlink()
        valid_ids_filepath.unlink()
        test_ids_filepath.unlink()


def preprocess(
    base_dir: Path,
    function_filepath: Path,
    train_id_filepath: Path,
    valid_id_filepath: Path,
    test_id_filepath: Path,
):
    """
    JV: Taken directly from CodeXGlue `preprocess.py`, modified to support a base directory.
    https://github.com/microsoft/CodeXGLUE/blob/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Defect-detection/dataset/preprocess.py
    """

    import json

    with open(function_filepath) as f:
        js_all = json.load(f)

    train_index = set()
    valid_index = set()
    test_index = set()

    with open(train_id_filepath) as f:
        for line in f:
            line = line.strip()
            train_index.add(int(line))

    with open(valid_id_filepath) as f:
        for line in f:
            line = line.strip()
            valid_index.add(int(line))

    with open(test_id_filepath) as f:
        for line in f:
            line = line.strip()
            test_index.add(int(line))

    filepath = str(base_dir / "train.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for idx, js in enumerate(js_all):
            if idx in train_index:
                js["idx"] = idx
                f.write(json.dumps(js) + "\n")

    filepath = str(base_dir / "valid.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for idx, js in enumerate(js_all):
            if idx in valid_index:
                js["idx"] = idx
                f.write(json.dumps(js) + "\n")

    filepath = str(base_dir / "test.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for idx, js in enumerate(js_all):
            if idx in test_index:
                js["idx"] = idx
                f.write(json.dumps(js) + "\n")


def collate_function(samples: list[dict]):
    def merge(key):
        return [s[key] for s in samples]

    pad_idx = get_pad_id()
    input_ids = collate_tokens(merge("tokens"), pad_idx=pad_idx, left_pad=False)

    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_idx),
        "labels": torch.tensor(merge("target"), dtype=torch.long),
        "idx": merge("idx"),
    }


class DefectsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset",
        batch_size: int = 32,
        max_length: Optional[int] = None,
        num_processes: Optional[int] = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)

        self.batch_size = batch_size
        self.max_length = max_length
        self.num_processes = num_processes

        self.dataset = None  # will be set by setup

    def prepare_data(self):
        if not (self.data_dir / "train.jsonl").exists():
            log.info(f"Downloading dataset to {self.data_dir}")
            download_dataset(data_dir=self.data_dir)

        load_and_preprocess_dataset(
            self.data_dir, self.max_length, num_proc=self.num_processes
        )  # ignore output

    def setup(self, stage=None):
        self.dataset = load_and_preprocess_dataset(
            self.data_dir, self.max_length, num_proc=self.num_processes
        )  # always load full dataset (it is tiny)

        self.dataset.set_format(
            type="torch", columns=["tokens"], output_all_columns=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_function,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=collate_function,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            collate_fn=collate_function,
            num_workers=1,
        )


if __name__ == "__main__":
    from cocos.utils import get_project_root

    dm = DefectsDataModule(
        data_dir=get_project_root()
        / "experiments/03-defect_detection_(defects)/dataset",
        num_processes=1,
    )
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)
        break
