import logging
from typing import List, Optional, Any, Iterable

import pytorch_lightning as pl

import torch
from pathlib import Path
import numpy as np


log = logging.getLogger(__name__)


def calc_scores(embeddings, normalize=True):
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)

    scores = embeddings @ embeddings.T
    scores.fill_diagonal_(-float("inf"))
    return scores


def evaluate_map_xglue(scores, labels):
    """Taken directly from codexglue and adapted. scoring is not changed
    embeddings: shape [S, s]
    labels: shape [S]
    ids: shape [S]

    S = number of samples
    D = dimensionality
    """
    import numpy as np

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind="quicksort", order=None)[:, ::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append((sum(Avep) / dic[label]) if dic[label] > 0 else 0)

    return np.mean(MAP).item(), scores, sort_ids


# taken from IBMCodeNet:
# https://github.com/IBM/Project_CodeNet/blob/c8047a379f868b20ddb6c2a61d1870d115b8ed63/model-experiments/token-based-similarity-classification/src/PostProcessor/MapAtR.py#L19-L53
# computes the same as evaluate_map_xglue from CodeXGlue
def map_at_r(scores, labels):
    """
    Function for computing MAP at R metric. R is set to number of solutions per problem.

    Parameter:
    - scores   --  2D numpy array of predicted similarity measures
                for all pairs of samples
    - labels  --  1D numpy array of problem ids corresponding
                to columns of matrix  of predicted similarity measures
    Returns: computed MAP at R metric
    """

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Count number of source code solutions of each problem
    r = np.bincount(labels) - 1
    max_r = r.max()
    # Mask for ignoring the similarity predictions lying
    # beyond the number of solutions of checked problem
    mask = np.arange(max_r)[np.newaxis, :] < r[labels][:, np.newaxis]
    np.fill_diagonal(scores, -np.inf)
    # Select and sort top predictions
    result = np.argpartition(-scores, range(max_r + 1), axis=1)[:, :max_r]
    # Get correct similarity predictions
    tp = labels[result] == labels[:, np.newaxis]
    # Remove all predictions beyond the number of
    # solutions of tested problem
    tp[~mask] = False
    # Get only tested problem
    valid = r[labels] > 0
    # Compute cumulative probability of correct predictions
    p = (
        np.cumsum(tp, axis=1, dtype=np.float32)
        / np.arange(1, max_r + 1, dtype=np.float32)[np.newaxis, :]
    )
    # average across similarity prediction for each tested problem
    ap = (p * tp).sum(axis=1)[valid] / r[labels][valid]
    val = np.mean(ap).item()
    return val


def predict_xglue(sort_ids, ids):
    """Taken directly from codexglue and adapted. scoring is not changed
    scores: shape [S, S]
    ids: shape [S]

    S = number of samples
    D = dimensionality
    """

    if isinstance(sort_ids, torch.Tensor):
        sort_ids = sort_ids.cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()

    for index, sort_id in zip(ids, sort_ids):
        js = {}
        js["index"] = str(int(index))
        js["answers"] = []
        for idx in sort_id[:499]:  # fixme this works only for defects detection
            js["answers"].append(str(int(ids[int(idx)])))
        yield js


def write_predictions(filepath, predictions: Iterable[dict]):
    import json

    p = Path(filepath)
    with p.open("wt") as f:
        lines = [json.dumps(pred) + "\n" for pred in predictions]
        f.writelines(lines)
    log.info(f"Wrote predictions to {p.absolute()}")


class RetrievalMetricsCallback(pl.Callback):
    """
    Note this works only single GPU.

    Models need to return dictionary with "embeddings" key in each step. These
    embeddings are gathered and at the end a similarity search is done.
    """

    def __init__(
        self,
        valid_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        normalize_embeddings: bool = True,
        compute_for_train_set: bool = False,
    ):
        super().__init__()
        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs = []
        self.compute_for_train_set = compute_for_train_set
        self.valid_filepath = valid_filepath
        self.test_filepath = test_filepath
        self.normalize_embeddings = normalize_embeddings

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.compute_for_train_set:
            if "labels" not in outputs:
                outputs["labels"] = batch["labels"].cpu()
            if "index" not in outputs:
                outputs["index"] = batch["index"].cpu()
            self.train_outputs.append({k: v.detach().cpu() for k, v in outputs.items()})

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if "labels" not in outputs:
            outputs["labels"] = batch["labels"].cpu()
        if "index" not in outputs:
            outputs["index"] = batch["index"].cpu()
        self.valid_outputs.append({k: v.detach().cpu() for k, v in outputs.items()})

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        output,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if "labels" not in output:
            output["labels"] = batch["labels"].cpu()
        if "index" not in output:
            output["index"] = batch["index"].cpu()
        self.test_outputs.append({k: v.detach().cpu() for k, v in output.items()})

    @staticmethod
    def gather_outputs(outputs: list[dict]):
        """Performs CPU"""
        embeddings, labels, ids = [], [], []

        for batch_out in outputs:
            embeddings.append(batch_out["embeddings"].cpu())
            labels.append(batch_out["labels"].cpu())
            ids.append(batch_out["index"].cpu())

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        ids = torch.cat(ids, dim=0)

        try:
            assert len(embeddings) == len(labels) == len(ids)
        except AssertionError as e:
            print(e)
            print("len(embeddings)", len(embeddings))
            print("len(labels)", len(labels))
            print("len(ids)", len(ids))
            raise e

        return embeddings, labels, ids

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.train_outputs = []

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.compute_for_train_set:
            embeddings, labels, ids = self.gather_outputs(self.train_outputs)

            scores = calc_scores(embeddings, self.normalize_embeddings)
            pl_module.log(
                "experiments/MAP",
                map_at_r(scores, labels),
                prog_bar=True,
                on_epoch=True,
            )

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.valid_outputs = []

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        embeddings, labels, ids = self.gather_outputs(self.valid_outputs)
        self.valid_outputs = []

        scores = calc_scores(embeddings, self.normalize_embeddings)
        pl_module.log(
            "valid/MAP", map_at_r(scores, labels), prog_bar=True, on_epoch=True
        )

        if self.valid_filepath is not None:
            sort_ids = torch.argsort(scores, dim=1, descending=True)
            write_predictions(self.valid_filepath, predict_xglue(sort_ids, ids))

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.test_outputs = []

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        embeddings, labels, ids = self.gather_outputs(self.test_outputs)
        self.test_outputs = []

        scores = calc_scores(embeddings, self.normalize_embeddings)
        pl_module.log(
            "test/MAP", map_at_r(scores, labels), prog_bar=True, on_epoch=True
        )

        if self.test_filepath is not None:
            sort_ids = torch.argsort(scores, dim=1, descending=True)
            write_predictions(self.test_filepath, predict_xglue(sort_ids, ids))
