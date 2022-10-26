import logging
from typing import List, Optional

import pytorch_lightning as pl

import torch
from torch import nn
from torch.optim import AdamW
import torchmetrics
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import get_polynomial_decay_schedule_with_warmup


log = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
        use_pooler: bool,
    ):
        super().__init__()
        if use_pooler:
            self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.use_pooler = use_pooler

    def forward(self, hidden_states: torch.Tensor):
        if self.use_pooler:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.dense(hidden_states)
            hidden_states = torch.tanh(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class PairClassificationHead(nn.Module):
    """Head for sentence-pair classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(2 * input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1) * 2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


from transformers import T5EncoderModel


class EncoderForSequenceClassification(pl.LightningModule):
    def __init__(
        self,
        encoder: T5EncoderModel,
        num_classes: int,
        learning_rate: float,
        warmup_percentage: float = 0.06,
        pooler_dim: Optional[int] = None,
        pooler_dropout: Optional[int] = None,
        use_eos_instead_of_cls: bool = True,
        average: str = "micro",
        measure_binary: bool = False,
        use_pooler: bool = True,
    ):
        super().__init__()
        self.model = encoder

        self.config = self.model.config
        self.num_classes = num_classes
        self.pooler_dim = self.config.d_model if pooler_dim is None else pooler_dim
        self.pooler_dropout = (
            self.config.dropout_rate if pooler_dropout is None else pooler_dropout
        )
        self.learning_rate = learning_rate
        self.warmup_percentage = warmup_percentage
        self.use_eos_instead_of_cls = use_eos_instead_of_cls
        self.classification_head = ClassificationHead(
            self.config.d_model,
            self.pooler_dim,
            self.num_classes,
            self.pooler_dropout,
            use_pooler=use_pooler,
        )
        self.model._init_weights(
            self.classification_head
        )  # initialize weights of classification head the same way as in model
        self.is_binary = measure_binary

        self.metrics = nn.ModuleDict(
            {
                "train_metrics": nn.ModuleDict(
                    {
                        "accuracy": torchmetrics.Accuracy(
                            num_classes=num_classes, average=average
                        ),
                        "precision": torchmetrics.Precision(
                            num_classes=num_classes, average=average
                        ),
                        "recall": torchmetrics.Recall(
                            num_classes=num_classes, average=average
                        ),
                        "f1": torchmetrics.F1Score(
                            num_classes=num_classes, average=average
                        ),
                    }
                ),
                "valid_metrics": nn.ModuleDict(
                    {
                        "accuracy": torchmetrics.Accuracy(
                            num_classes=num_classes, average=average
                        ),
                        "precision": torchmetrics.Precision(
                            num_classes=num_classes, average=average
                        ),
                        "recall": torchmetrics.Recall(
                            num_classes=num_classes, average=average
                        ),
                        "f1": torchmetrics.F1Score(
                            num_classes=num_classes, average=average
                        ),
                    }
                ),
                "test_metrics": nn.ModuleDict(
                    {
                        "accuracy": torchmetrics.Accuracy(
                            num_classes=num_classes, average=average
                        ),
                        "precision": torchmetrics.Precision(
                            num_classes=num_classes, average=average
                        ),
                        "recall": torchmetrics.Recall(
                            num_classes=num_classes, average=average
                        ),
                        "f1": torchmetrics.F1Score(
                            num_classes=num_classes, average=average
                        ),
                    }
                ),
            }
        )

        if measure_binary:
            for split, metric_dict in self.metrics.items():
                metric_dict["accuracy (binary)"] = torchmetrics.Accuracy(
                    num_classes=1, multiclass=False
                )
                metric_dict["precision (binary)"] = torchmetrics.Precision(
                    num_classes=1, multiclass=False
                )
                metric_dict["recall (binary)"] = torchmetrics.Recall(
                    num_classes=1, multiclass=False
                )
                metric_dict["f1 (binary)"] = torchmetrics.F1Score(
                    num_classes=1, multiclass=False
                )

    @property
    def batch_size(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            return dm.batch_size

    @property
    def validation_batch_size(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            return getattr(dm, "validation_batch_size", self.batch_size)

    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs[0]  # last layer hidden state

        # fixme use EOS or CLS token??
        # eos
        if self.use_eos_instead_of_cls:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )
            sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
            )[:, -1, :]
        else:
            # CLS
            sentence_representation = hidden_states[:, 0]

        return sentence_representation, outputs

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        sentence_representation, outputs = self.embed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_classes == 1:
                    self.config.problem_type = "regression"
                elif self.num_classes > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self.forward(**batch)
        loss = output.loss

        preds = output.logits.argmax(-1)

        for metricname, metric in self.metrics["train_metrics"].items():
            metric(preds, batch["labels"])
            self.log(
                f"train/{metricname}",
                metric,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        output = self.forward(**batch)
        loss = output.loss
        preds = output.logits.argmax(-1)
        for metricname, metric in self.metrics["valid_metrics"].items():
            metric(preds, batch["labels"])
            self.log(
                f"valid/{metricname}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.validation_batch_size,
            )

        self.log(
            "valid/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.validation_batch_size,
        )

        keys_to_output = [
            k for k in batch if k != "input_ids" and k != "attention_mask"
        ]
        outputs = {
            k: batch[k].squeeze().tolist()
            if isinstance(batch[k], torch.Tensor)
            else batch[k]
            for k in keys_to_output
        }
        outputs["prediction"] = preds.squeeze().tolist()
        return {"loss": loss, "outputs": outputs}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        output = self.forward(**batch)

        loss = output.loss

        preds = output.logits.argmax(-1)

        for metricname, metric in self.metrics["test_metrics"].items():
            metric(preds, batch["labels"])
            self.log(
                f"test/{metricname}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.validation_batch_size,
            )

        keys_to_output = [
            k for k in batch if k != "input_ids" and k != "attention_mask"
        ]
        outputs = {
            k: batch[k].squeeze().tolist()
            if isinstance(batch[k], torch.Tensor)
            else batch[k]
            for k in keys_to_output
        }
        outputs["prediction"] = preds.squeeze().tolist()
        return {"loss": loss, "outputs": outputs}

    def approx_number_of_steps(self):
        if self.trainer.max_steps > -1 and self.trainer.max_steps is not None:
            return self.trainer.max_steps
        elif self.trainer.max_epochs is not None:
            num_train_samples = len(self.trainer.datamodule.dataset["train"])

            import math

            approx_num_batches_per_epoch = math.ceil(
                num_train_samples // self.trainer.datamodule.batch_size
            )

            if self.trainer.limit_train_batches > 1:
                approx_num_batches_per_epoch = self.trainer.limit_train_batches
            else:
                approx_num_batches_per_epoch *= self.trainer.limit_train_batches

            print("approx_num_batches_per_epoch", approx_num_batches_per_epoch)
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


# FIXME remove??
# class EncoderForSequencePairClassification(EncoderForSequenceClassification):
#     def __init__(self, encoder: T5EncoderModel, num_classes: int, learning_rate: float, warmup_percentage: float = 0.06,
#                  pooler_dim: Optional[int] = None, pooler_dropout: Optional[int] = None, use_eos_instead_of_cls: bool = True,
#                  average: str = "micro", measure_binary: bool = False):
#         super(EncoderForSequencePairClassification, self).__init__(
#             encoder=encoder,
#             num_classes=num_classes,
#             learning_rate=learning_rate,
#             warmup_percentage=warmup_percentage,
#             pooler_dim=pooler_dim,
#             pooler_dropout=pooler_dropout,
#             use_eos_instead_of_cls=use_eos_instead_of_cls,
#             average=average,
#             measure_binary=measure_binary
#         )
#         self.classification_head = PairClassificationHead(
#             self.config.d_model, self.pooler_dim, self.num_classes, self.pooler_dropout
#         )
#         self.model._init_weights(self.classification_head)
#
#         self.cache = {}
#
#     def forward(
#             self,
#             input_ids=None,
#             labels=None,
#             attention_mask=None,
#             head_mask=None,
#             inputs_embeds=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             use_cache=False,
#             **kwargs
#     ):
#         if not use_cache:
#             sentence_representations, _ = self.embed(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )
#         else:
#             assert "idx" in kwargs
#             ids = kwargs["idx"]
#
#             B, N = input_ids.shape
#
#             mask = torch.tensor([True if id_ in self.cache else False for id_ in kwargs["idx"]])
#             ids_in_index = mask.nonzero().view(-1)
#             ids_not_in_index = (~mask).nonzero().view(-1)
#
#             sentence_representations = input_ids.new_zeros((B, self.config.d_model), dtype=torch.float).to(self.device)
#             # print("sentence_representations.shape", sentence_representations.shape)
#
#             if ids_in_index.numel() > 0:
#                 sentence_representation_cached = torch.stack(
#                     [self.cache[ids[idx.item()]] for idx in ids_in_index], dim=0
#                 ).to(self.device)
#                 # print("sentence_representation_cached.shape", sentence_representation_cached.shape)
#                 sentence_representations[ids_in_index] = sentence_representation_cached
#
#             if ids_not_in_index.numel() > 0:
#                 ips = input_ids[ids_not_in_index]
#                 ips = ips if ips.ndim == 2 else ips[None,:]
#                 am = attention_mask[ids_not_in_index]
#                 am = am if am.ndim == 2 else am[None,:]
#                 # print("ips", ips.shape)
#                 # print("am", am.shape)
#                 sentence_representation_not_cached, _ = self.embed(
#                     input_ids=ips,
#                     attention_mask=am,
#                 )
#                 # print("ids_not_in_index", ids_not_in_index)
#                 # print("ids_not_in_index.shape", ids_not_in_index.shape)
#                 # print("sentence_representation_not_cached.shape", sentence_representation_not_cached.shape)
#                 sentence_representations[ids_not_in_index] = sentence_representation_not_cached
#
#                 # save in cache
#                 for idx, embedding in zip(ids_not_in_index, sentence_representation_not_cached):
#                     self.cache[ids[idx.item()]] = embedding.squeeze().detach().cpu()
#
#         # print(self.cache)
#         logits = self.classification_head(sentence_representations)
#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_classes == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = nn.MSELoss()
#                 if self.num_classes == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = nn.BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             # hidden_states=outputs.hidden_states,
#             # attentions=outputs.attentions,
#         )
#
#
#     def validation_step(self, batch: dict, batch_idx: int) -> dict:
#         output = self.forward(**batch, use_cache=True)
#         loss = output.loss
#
#         preds = output.logits.argmax(-1)
#
#         for metricname, metric in self.metrics["valid_metrics"].items():
#             metric(preds, batch["labels"])
#             self.log(f'valid/{metricname}', metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.validation_batch_size)
#
#         self.log('valid/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.validation_batch_size)
#
#         keys_to_output = [k for k in batch if k != "input_ids" and k != "attention_mask"]
#         outputs = {k: batch[k].squeeze().tolist() if isinstance(batch[k], torch.Tensor) else batch[k] for k in keys_to_output}
#         outputs["prediction"] = preds.squeeze().tolist()
#         return {
#             "loss": loss,
#             "outputs": outputs
#         }
#
#     def on_validation_epoch_start(self) -> None:
#         self.cache = {}
#
#     def on_validation_epoch_end(self) -> None:
#         self.cache = {}
#
#     def on_test_epoch_start(self) -> None:
#         self.cache = {}
#
#     def on_test_epoch_end(self) -> None:
#         self.cache = {}
#
#     def test_step(self, batch: dict, batch_idx: int) -> dict:
#         output = self.forward(**batch, use_cache=True)
#
#         loss = output.loss
#
#         preds = output.logits.argmax(-1)
#
#         for metricname, metric in self.metrics["test_metrics"].items():
#             metric(preds, batch["labels"])
#             self.log(f'test/{metricname}', metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.validation_batch_size)
#
#         keys_to_output = [k for k in batch if k != "input_ids" and k != "attention_mask"]
#         outputs = {k: batch[k].squeeze().tolist() if isinstance(batch[k], torch.Tensor) else batch[k] for k in keys_to_output}
#         outputs["prediction"] = preds.squeeze().tolist()
#         return {
#             "loss": loss,
#             "outputs": outputs
#         }
