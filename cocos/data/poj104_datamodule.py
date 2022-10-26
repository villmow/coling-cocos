import logging
import shutil
from pathlib import Path
from typing import List, Optional

import datasets
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_metric_learning.samplers import MPerClassSampler
import torch
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path

from cocos.tensortree.collate import collate_tokens
from cocos.tokenizer import (
    get_language_token_id,
    get_eos_id,
    get_pad_id,
    CodeTokenizer,
    load_tokenizer,
)


log = logging.getLogger(__name__)


def encode_sample(
    sample,
    tokenizer: CodeTokenizer,
    c_lang_id_token: int,
    cpp_lang_id_token: int,
    eos_id: int,
    max_length=None,
):
    code = sample["code"]

    lang = "c"
    # we dont know if this code uses c or c++
    cpp_out = tokenizer.parse(code, "cpp")
    if cpp_out.num_errors > 0:
        c_out = tokenizer.parse(code, "c")
        if cpp_out.num_errors < c_out.num_errors:
            out = cpp_out
            lang = "cpp"
        else:
            out = c_out
    else:
        out = cpp_out
        lang = "cpp"

    tree = out.tree
    encoding = tokenizer.encode(tree)

    tokens = encoding.ids
    if max_length is not None:
        tokens = tokens[: max_length - 2]

    tokens = [c_lang_id_token if lang == "c" else cpp_lang_id_token] + tokens + [eos_id]
    return {
        "tokens": tokens,
        "label": int(sample["label"]),
        "index": int(sample["index"]),
    }


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
    return dataset.map(
        encode_sample,
        batched=False,
        num_proc=num_proc,
        fn_kwargs={
            "tokenizer": load_tokenizer(),
            "c_lang_id_token": get_language_token_id("c"),
            "cpp_lang_id_token": get_language_token_id("cpp"),
            "eos_id": get_eos_id(),
            "max_length": max_length,
        },
    )


def collate_function(samples: list[dict]):
    def merge(key):
        return [s[key] for s in samples]

    # print(samples)
    pad_idx = get_pad_id()
    input_ids = collate_tokens(merge("tokens"), pad_idx=pad_idx, left_pad=False)
    return {
        "index": torch.tensor(merge("index"), dtype=torch.long),
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_idx),
        "labels": torch.tensor(merge("label"), dtype=torch.long),
    }


def download_dataset(data_dir: Path, keep_intermediate_files: bool = False):
    """Downloads and extracts the dataset into data_dir."""
    import gdown

    url = "https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU"

    data_dir = Path(to_absolute_path(str(data_dir)))
    data_dir.mkdir(parents=True, exist_ok=True)

    outfile = data_dir / "programs.tar.gz"

    if not outfile.exists():
        gdown.download(url, str(outfile), quiet=False)

    extracted_directory = data_dir / "ProgramData"
    if not extracted_directory.exists():
        import tarfile

        with tarfile.open(outfile) as f:
            # extracting file
            f.extractall(data_dir)

    preprocess(data_dir)

    if not keep_intermediate_files:
        # cleanup files
        outfile.unlink()
        shutil.rmtree(extracted_directory)


def preprocess(base_dir: str):
    """
    JV: Taken directly from CodeXGlue `preprocess.py`, modified to support a base directory.
    https://github.com/microsoft/CodeXGLUE/blob/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Clone-detection-POJ-104/dataset/preprocess.py
    """
    base_dir = Path(to_absolute_path(str(base_dir)))

    # Copyright (c) Microsoft Corporation.
    # Licensed under the MIT License.

    import os
    import json
    from tqdm import tqdm

    def files(path):
        path = str(base_dir / path)  # JV: added
        g = os.walk(path)
        file = []
        for path, dir_list, file_list in g:
            for file_name in file_list:
                file.append(os.path.join(path, file_name))
        return file

    cont = 0

    filepath = str(base_dir / "train.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for i in tqdm(range(1, 65), total=64):
            items = files("ProgramData/{}".format(i))
            for item in items:
                js = {}
                js["label"] = item.split("/")[-2]  # jv: modified 1 to -2
                js["index"] = str(cont)
                js["code"] = open(item, encoding="latin-1").read()
                f.write(json.dumps(js) + "\n")
                cont += 1
    filepath = str(base_dir / "valid.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for i in tqdm(range(65, 81), total=16):
            items = files("ProgramData/{}".format(i))
            for item in items:
                js = {}
                js["label"] = item.split("/")[-2]  # jv: modified 1 to -2
                js["index"] = str(cont)
                js["code"] = open(item, encoding="latin-1").read()
                f.write(json.dumps(js) + "\n")
                cont += 1
    filepath = str(base_dir / "test.jsonl")  # JV: added
    with open(filepath, "w") as f:
        for i in tqdm(range(81, 195), total=24):
            items = files("ProgramData/{}".format(i))
            for item in items:
                js = {}
                js["label"] = item.split("/")[-2]  # jv: modified 1 to -2
                js["index"] = str(cont)
                js["code"] = open(item, encoding="latin-1").read()
                f.write(json.dumps(js) + "\n")
                cont += 1


class POJ104DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset",
        batch_size: int = 32,
        max_length: Optional[int] = None,
        num_processes: Optional[int] = 1,
        n_samples_of_class_per_batch: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_processes = num_processes
        self.n_samples_of_class_per_batch = n_samples_of_class_per_batch

        self.dataset = None  # will be set by setup

    def prepare_data(self):
        if not (self.data_dir / "train.jsonl").exists():
            log.info(f"Downloading dataset to {self.data_dir}")
            download_dataset(data_dir=self.data_dir)

        load_and_preprocess_dataset(
            self.data_dir, self.max_length, num_proc=self.num_processes
        )  # always load full dataset (it is tiny)

    def setup(self, stage=None):
        self.dataset = load_and_preprocess_dataset(
            self.data_dir, self.max_length, num_proc=self.num_processes
        )  # always load full dataset (it is tiny)
        self.dataset.set_format(
            type="torch", columns=["tokens", "label", "index"], output_all_columns=False
        )
        # self.dataset = encoded_dataset

    def train_dataloader(self):
        shuffle = True
        sampler = None
        if self.n_samples_of_class_per_batch is not None:
            shuffle = False

            from torch.utils.data import Sampler

            class FixIndices(Sampler):
                """
                MPerClassSampler sampler returns numpy indices which do not work with huggingface dataset.
                So this sampler simply turns them into integers.
                """

                def __init__(self, original_sampler: Sampler):
                    super().__init__(original_sampler)
                    self.original_sampler = original_sampler

                def __len__(self):
                    return len(self.original_sampler)

                def __iter__(self):
                    for x in self.original_sampler:
                        yield int(x)

            sampler = MPerClassSampler(
                labels=self.dataset["experiments"]["label"],
                m=self.n_samples_of_class_per_batch,
                batch_size=self.batch_size,
            )
            sampler = FixIndices(sampler)

        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_function,
            num_workers=1,
            sampler=sampler,
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

    dm = POJ104DataModule(
        data_dir=get_project_root() / "experiments/04-clone_detection_(poj104)/dataset",
        num_processes=1,
    )
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)
        break
