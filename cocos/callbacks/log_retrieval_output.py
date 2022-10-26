import logging
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from transformers.models.rag.modeling_rag import RetrievAugLMOutput

from cocos import tokenizer


log = logging.getLogger(__file__)


class LogRetrievalCallback(pl.Callback):
    def __init__(
        self,
        log_first_n_training_batches: int = 4,
        log_first_n_validation_batches: int = 0,
        log_first_n_test_batches: int = 0,
        log_first_n_predict_batches: int = 0,
        log_tensors: bool = False,
    ):
        super(LogRetrievalCallback, self).__init__()
        self.tokenizer = tokenizer.load_tokenizer()
        self.pad_id = tokenizer.get_pad_id()

        self.log_first_n_training_batches = log_first_n_training_batches
        self.log_first_n_validation_batches = log_first_n_validation_batches
        self.log_first_n_test_batches = log_first_n_test_batches
        self.log_first_n_predict_batches = log_first_n_predict_batches

        self.log_tensors = log_tensors

    def decode(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        if not lengths.ndim:
            lengths = lengths.unsqueeze(0)
        return (self.tokenizer.decode(t[:l]) for t, l in zip(token_ids, lengths))
        # self.tokenizer.tokenizer.tokenizer.batch_decode(sequences=token_ids) #, skip_special_tokens=True)

    def log_retrieval_results(
        self, output: RetrievAugLMOutput, batch: Any, stage: str = ""
    ):
        # try:
        query_strings = self.decode(batch.source.tokens, batch.source.lengths)
        target_strings = self.decode(batch.target.tokens, batch.target.lengths)
        doc_scores = output.doc_scores
        r_lengths = torch.sum(output.context_attention_mask, 2)
        retrieved_strings = [
            self.decode(retrieved, l)
            for retrieved, l in zip(output.context_input_ids, r_lengths)
        ]
        for query, target, score, retrieved in zip(
            query_strings, target_strings, doc_scores, retrieved_strings
        ):
            log.info("\n" + "~" * 50 + "\n" + "#" * 50)
            log.info(f"[{stage.upper()}] QUERY: \n {query}\n\n" + "~" * 50)
            log.info(f"[{stage.upper()}] TARGET: \n {target}\n\n" + "~" * 50)

            for s, r in zip(score, retrieved):
                log.info(f"[{stage.upper()}] RETRIEVED: score={s}\n {r}\n" + "-" * 50)

        # except Exception as e:
        #    log.error(f"[{stage.upper()}] [{self.__class__.__name__}] Exception while logging retrieval results: {repr(e)}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if (
            batch_idx < self.log_first_n_training_batches
            and isinstance(outputs, RetrievAugLMOutput)
            and outputs.retrieved_doc_ids
        ):
            log.info(f"Logging retrieval results for batch #{batch_idx}:")
            self.log_retrieval_results(outputs[0], batch, "experiments")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if (
            batch_idx < self.log_first_n_validation_batches
            and isinstance(outputs[0], RetrievAugLMOutput)
            and len(outputs[0].retrieved_doc_ids) > 0
        ):
            log.info(f"Logging retrieval results for batch #{batch_idx}:")
            self.log_retrieval_results(outputs[0], batch, "valid")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if (
            batch_idx < self.log_first_n_validation_batches
            and isinstance(outputs, RetrievAugLMOutput)
            and outputs.retrieved_doc_ids
        ):
            log.info(f"Logging retrieval results for batch #{batch_idx}:")
            self.log_retrieval_results(outputs, batch_idx, "test")
        if isinstance(outputs, dict) and "outputs" in outputs:
            self.append_outputs_to_file(outputs["outputs"], self.test_filename)
