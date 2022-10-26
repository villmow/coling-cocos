import logging
from typing import List, Optional, Any, Iterable

import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners, reducers, distances

from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import T5EncoderModel
import torch
from torch.optim import AdamW


log = logging.getLogger(__name__)


class EncoderForRetrieval(pl.LightningModule):
    """Finetunes a T5Encoder model on a retrieval task (such as clone detection")."""

    def __init__(
        self,
        encoder: T5EncoderModel,
        loss: losses.BaseMetricLossFunction,
        learning_rate: float,
        warmup_percentage: float = 0.06,
        use_eos_instead_of_cls: bool = False,
        miner: Optional[miners.BaseMiner] = None,
    ):
        super().__init__()

        self.model = encoder
        self.config = self.model.config
        self.learning_rate = learning_rate
        self.warmup_percentage = warmup_percentage
        self.use_eos_instead_of_cls = use_eos_instead_of_cls
        self.loss = loss
        self.miner = miner

    @classmethod
    def load_from_checkpoint_and_config(
        cls, checkpoint_path, config_path, config_overrides: Optional[dict] = None
    ):
        from cocos.utils import load_config, get_project_root
        import hydra
        from omegaconf import OmegaConf

        cfg = load_config(str(config_path))

        if config_overrides is not None:
            for key, value in config_overrides.items():
                OmegaConf.update(cfg, key, value)

        obj: EncoderForRetrieval = hydra.utils.instantiate(cfg.model)
        return EncoderForRetrieval.load_from_checkpoint(
            checkpoint_path, model=obj.model
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # last layer hidden state

        if self.use_eos_instead_of_cls:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )
            embeddings = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
            )[:, -1, :]
        else:
            embeddings = hidden_states[:, 0]

        return embeddings

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        labels = batch["labels"]
        embeddings = self.forward(**batch)

        if self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss(embeddings, labels, hard_pairs)
        else:
            loss = self.loss(embeddings, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return {
            "loss": loss,
            "embeddings": embeddings.detach(),
        }

    def validation_step(self, batch: dict, batch_idx, dataloader_idx=0) -> dict:
        embeddings = self.forward(**batch)

        if self.loss is not None:
            loss = self.loss(embeddings, batch["labels"])
            self.log("valid/loss", loss, on_step=True, on_epoch=True)
        else:
            loss = None

        return {
            "loss": loss,
            "embeddings": embeddings.detach(),
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        embeddings = self.forward(**batch)

        if self.loss is not None:
            loss = self.loss(embeddings, batch["labels"])
            self.log("test/loss", loss, on_step=True, on_epoch=True)
        else:
            loss = None

        return {
            "loss": loss,
            "embeddings": embeddings.detach(),
        }

    def approx_number_of_steps(self):
        if self.trainer.max_steps > -1 and self.trainer.max_steps is not None:
            return self.trainer.max_steps
        elif self.trainer.max_epochs is not None:
            num_train_samples = len(self.trainer.datamodule.dataset["train"])
            import math

            approx_num_batches_per_epoch = math.ceil(
                num_train_samples // self.trainer.datamodule.batch_size
            )
            return self.trainer.max_epochs * approx_num_batches_per_epoch
        else:
            raise ValueError("Somthing must be set")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        num_steps = self.approx_number_of_steps()
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_steps * self.warmup_percentage),
            num_training_steps=num_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
