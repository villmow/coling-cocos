from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

import hydra.utils
from torch.optim import AdamW
import pytorch_lightning as pl
from omegaconf import DictConfig

import logging
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from cocos.source_code import BatchedCodePairs
from cocos.tokenizer import get_pad_id
from pytorch_metric_learning import losses


log = logging.getLogger(__file__)


def load_model(
    checkpoint_path,
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[DictConfig] = None,
) -> pl.LightningModule:
    from hydra.utils import to_absolute_path

    if config is None:
        from cocos.utils import load_config

        config = load_config(to_absolute_path(str(config_path)))

    model_cls = hydra.utils.get_class(config.model._target_)
    log.info(f"Loading pretrained model of type {model_cls}")

    kwargs = {
        k: hydra.utils.instantiate(v)
        if isinstance(v, DictConfig) and "_target_" in v
        else v
        for k, v in config.model.items()
    }

    model = model_cls.load_from_checkpoint(
        to_absolute_path(str(checkpoint_path)), **kwargs
    )

    return model


class Retriever(pl.LightningModule):
    def __init__(self, loss, optimizer_cfg):
        super().__init__()

        self.loss_func = loss
        self.optimizer_cfg = optimizer_cfg
        self.pad_id = get_pad_id()

    def create_pair_labels(self, num_pos_pairs) -> Union[dict, torch.Tensor]:
        labels = torch.arange(0, num_pos_pairs).to(self.device).repeat(2)

        if isinstance(self.loss_func, losses.CrossBatchMemory):
            return self._to_cross_batch_pair_labels(labels)

        return labels

    def _to_cross_batch_pair_labels(self, labels: torch.Tensor) -> dict:
        assert labels.size(0) % 2 == 0
        num_pos_pairs = int(labels.size(0) / 2)

        previous_max_label = torch.max(self.loss_func.label_memory)
        # add an offset so that the labels do not overlap with any labels in the memory queue
        labels += previous_max_label + 1
        # we want to enqueue the output of encK, which is the 2nd half of the batch
        enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs * 2)
        return {"labels": labels, "enqueue_idx": enqueue_idx}

    def compute_pair_loss(self, source_embeddings, target_embeddings):
        if isinstance(self.loss_func, losses.VICRegLoss):
            # directly compare embeddings
            loss = self.loss_func(source_embeddings, target_embeddings)
        else:
            embeddings = torch.cat([source_embeddings, target_embeddings], dim=0)
            label_out = self.create_pair_labels(source_embeddings.size(0))
            if isinstance(label_out, dict):
                loss = self.loss_func(embeddings, **label_out)
            else:
                loss = self.loss_func(embeddings, label_out)

        return loss

    def forward_pair(self, batch: BatchedCodePairs):
        query_embeddings = self.forward_query(batch)
        key_embeddings = self.forward_key(batch)
        loss = self.compute_pair_loss(query_embeddings, key_embeddings)
        return loss, query_embeddings, key_embeddings

    def training_step(self, batch: BatchedCodePairs, batch_idx: int) -> torch.Tensor:
        loss, query_embeddings, key_embeddings = self.forward_pair(batch)
        bsz = batch.target.nsamples
        self.log(
            "train/batchsize",
            float(batch.target.nsamples),
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
        )
        self.log(
            "train/input_tokens",
            float(batch.source.ntokens),
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
        )
        self.log(
            "train/target_tokens",
            float(batch.target.ntokens),
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=bsz)

        return loss

    def validation_step(
        self, batch: BatchedCodePairs, batch_idx, dataloader_idx=0
    ) -> Dict:
        loss, query_embeddings, key_embeddings = self.forward_pair(batch)

        bsz = batch.target.nsamples
        self.log(
            "valid/batchsize",
            float(batch.target.nsamples),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "valid/input_tokens",
            float(batch.source.ntokens),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "valid/target_tokens",
            float(batch.target.ntokens),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "valid/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bsz,
        )

        return {
            "loss": loss,
            "queries": batch.source.tokens[batch.source.tokens != self.pad_id]
            .cpu()
            .tensor_split(batch.source.lengths.cumsum(-1).cpu())[:-1],
            "keys": batch.target.tokens[batch.target.tokens != self.pad_id]
            .cpu()
            .tensor_split(batch.target.lengths.cumsum(-1).cpu())[:-1],
            "queries_meta": batch.source.meta,
            "keys_meta": batch.target.meta,
            "query_embeddings": F.normalize(query_embeddings, dim=1).cpu(),
            "key_embeddings": F.normalize(key_embeddings, dim=1).cpu(),
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, query_embeddings, key_embeddings = self.forward_pair(batch)

        bsz = batch.target.nsamples
        self.log(
            "test/batchsize",
            float(batch.target.nsamples),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "test/input_tokens",
            float(batch.source.ntokens),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "test/target_tokens",
            float(batch.target.ntokens),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bsz,
        )

        return {
            "loss": loss,
            "queries": batch.source.tokens[batch.source.tokens != self.pad_id]
            .cpu()
            .tensor_split(batch.source.lengths.cumsum(-1).cpu())[:-1],
            "keys": batch.target.tokens[batch.target.tokens != self.pad_id]
            .cpu()
            .tensor_split(batch.target.lengths.cumsum(-1).cpu())[:-1],
            "queries_meta": batch.source.meta,
            "keys_meta": batch.target.meta,
            "query_embeddings": F.normalize(query_embeddings, dim=1).cpu(),
            "key_embeddings": F.normalize(key_embeddings, dim=1).cpu(),
        }

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.learning_rate,
            eps=self.optimizer_cfg.adam_epsilon,
        )

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.optimizer_cfg.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.total_steps = self.trainer.max_steps
            assert self.total_steps is not None
