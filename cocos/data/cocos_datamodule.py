import logging
from pathlib import Path
from typing import List, Optional

import datasets
from datasets import load_dataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cocos.tensortree.collate import collate_tokens
from cocos.tokenizer import (
    load_tokenizer,
    get_language_token_id,
    get_pad_id,
)
from cocos.source_code.indentation import indentation_level_before_index, dedent


log = logging.getLogger(__name__)


def encode(
    sample,
    language: str,
    max_length=None,
):
    code = sample["code"]
    tokenizer = load_tokenizer()
    input_ids = tokenizer.encode(code, language).ids

    original_length = len(input_ids)
    if max_length is not None:
        input_ids = input_ids[: max_length - 2]

    input_ids = [get_language_token_id(language)] + input_ids

    return {
        "input_ids": input_ids,
        "original_length": original_length,
    }


def load_cocos_dataset(
    data_filepath,
) -> tuple[datasets.Dataset, datasets.Dataset, list, list]:
    from hydra.utils import to_absolute_path

    data_filepath = Path(to_absolute_path(str(data_filepath)))

    dataset = load_dataset("json", data_files=str(data_filepath))["train"]

    # problemid 0 is for distraction snippets
    id2problem = ["distraction"] + sorted(list(set(dataset["problem"])))
    problem2id = {p: i for i, p in enumerate(id2problem)}

    id2description = [None] * len(id2problem)
    for sample in dataset:
        problemid = problem2id[sample["problem"]]
        id2description[problemid] = sample["problem_description"]

    # add query_id == idx to each sample
    dataset = dataset.map(
        lambda x, idx: {
            "query_id": idx,
            "label": problem2id[x["problem"]],
        },
        with_indices=True,
        batched=False,
    )

    query_dataset = dataset.select_columns(
        [
            "problem",
            "problem_description",
            # "target",
            "method_masked",
            # "context_masked",
            # "mask_start",
            # "mask_end",
            # "method_start",
            # "method_end",
            # "context_start",
            # "context_end",
            # "file",
            "query_id",
            "label",
        ]
    ).rename_column("method_masked", "code")
    target_dataset = dataset.select_columns(
        [
            "problem",
            "problem_description",
            "target",
            # "method_masked",
            # "context_masked",
            # "mask_start",
            # "mask_end",
            # "method_start",
            # "method_end",
            # "context_start",
            # "context_end",
            # "file",
            "query_id",
            "label",
        ]
    ).rename_column("target", "code")

    return query_dataset, target_dataset, id2problem, id2description


def load_distractor_dataset(
    data_filepath,
    max_distraction_snippets: int = 0,
) -> datasets.Dataset:
    from hydra.utils import to_absolute_path

    data_filepath = Path(to_absolute_path(str(data_filepath)))

    dataset = load_dataset("json", data_files=str(data_filepath))["train"]

    dataset = dataset.map(
        lambda x: {"label": 0, "query_id": -1},
        batched=False,
        with_indices=False,
    )

    dataset = dataset.select_columns(
        [
            # "repo",
            # "path",
            "language",
            # "sha",
            # "url",
            # "docstring",
            # "code_full",
            # "code_body",
            "code_sampled_span",
            "label",
            "query_id",
        ]
    ).rename_column("code_sampled_span", "code")

    # select first x rows
    if max_distraction_snippets:
        dataset = dataset.select(range(min(max_distraction_snippets, len(dataset))))

    return dataset


def collate_function(samples: list[dict]):
    def merge(key):
        return [s[key] for s in samples]

    pad_idx = get_pad_id()

    input_ids = collate_tokens(merge("input_ids"), pad_idx=pad_idx, left_pad=False)
    labels = torch.tensor(merge("label"), dtype=torch.long)

    return {
        "query_ids": torch.tensor(merge("query_id"), dtype=torch.long),
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_idx),
        "labels": labels,
    }


class CCSZSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_filepath: str = "dataset.jsonl",
        distraction_filepath: str = "distractors.jsonl",
        batch_size: int = 32,
        max_length: Optional[int] = None,
        num_processes: Optional[int] = 1,
        max_distraction_snippets: Optional[int] = None,
    ):
        super().__init__()
        self.data_filepath = data_filepath
        self.distraction_filepath = distraction_filepath
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_processes = num_processes
        self.max_distraction_snippets = max_distraction_snippets

        self.query_dataset = None  # will be set by setup
        self.target_dataset = None  # will be set by setup
        self.distractor_dataset = None  # will be set by setup
        self.id2problem = None  # will be set by setup
        self.id2description = None  # will be set by setup

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        (
            query_dataset,
            target_dataset,
            id2problem,
            id2description,
        ) = load_cocos_dataset(self.data_filepath)

        distractor_dataset = load_distractor_dataset(
            self.distraction_filepath,
            max_distraction_snippets=self.max_distraction_snippets,
        )

        if distractor_dataset:
            search_index_dataset = datasets.concatenate_datasets(
                [target_dataset, distractor_dataset]
            )
        else:
            search_index_dataset = target_dataset

        # remove leading whitespace from search index
        search_index_dataset = search_index_dataset.map(
            lambda x: {"code": x["code"].lstrip()},
            batched=False,
            with_indices=False,
            num_proc=self.num_processes,
        )

        # encode
        search_index_dataset = search_index_dataset.map(
            encode,
            batched=False,
            with_indices=False,
            num_proc=self.num_processes,
            fn_kwargs={
                "language": "java",
                "max_length": self.max_length,
            },
        )
        query_dataset = query_dataset.map(
            encode,
            batched=False,
            with_indices=False,
            num_proc=self.num_processes,
            fn_kwargs={
                "language": "java",
                "max_length": self.max_length,
            },
        )

        self.query_dataset = query_dataset
        self.target_dataset = target_dataset
        self.distractor_dataset = distractor_dataset
        self.search_index_dataset = search_index_dataset

        self.id2problem = id2problem
        self.id2description = id2description

        # self.query_dataset = self.query_dataset.sort("original_length")
        self.query_dataset.set_format(
            type="torch",
            columns=["input_ids", "label", "query_id"],
            output_all_columns=False,
        )

        # self.search_index_dataset = self.search_index_dataset.sort("original_length")
        self.search_index_dataset.set_format(
            type="torch",
            columns=["input_ids", "label", "query_id"],
            output_all_columns=False,
        )

        self.query_dataset = self.query_dataset.sort("original_length")
        self.search_index_dataset = self.search_index_dataset.sort("original_length")

    def train_dataloader(self):
        raise NotImplementedError("Zeroshot Dataset")

    def val_dataloader(self):
        query_dataloader = DataLoader(
            self.query_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_function,
            num_workers=1,
            shuffle=False,
        )
        search_index_dataloader = DataLoader(
            self.search_index_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_function,
            num_workers=1,
            shuffle=False,
        )
        return query_dataloader, search_index_dataloader

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    datasets.disable_caching()

    ds = CCSZSDataModule(
        data_filepath="../../dataset/cocos.jsonl",
        distraction_filepath="../../dataset/distractors.jsonl",
        max_length=None,
        num_processes=1,
        batch_size=1,
        max_distraction_snippets=100,
    )
    ds.prepare_data()
    ds.setup()
    tokenizer = load_tokenizer()

    query_dataloader, search_index_dataloader = ds.val_dataloader()
    for batch in query_dataloader:
        print("#" * 100)
        print(tokenizer.decode(batch["input_ids"].squeeze()))
        print([ds.id2problem[label.item()] for label in batch["labels"]])
        break

    print(tokenizer.decode(ds.search_index_dataset[0]["input_ids"]))
    print([ds.id2problem[label.item()] for label in batch["labels"]])

    for batch in search_index_dataloader:
        print("#" * 100)
        print(tokenizer.decode(batch["input_ids"].squeeze()))
        print([ds.id2problem[label.item()] for label in batch["labels"]])
        break

    print(ds.query_dataset)
    print(ds.search_index_dataset)
