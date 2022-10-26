import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

import pytorch_lightning as pl

from ktz.filesystem import path_rotate

import json


log = logging.getLogger(__file__)


class OutputSaver(pl.Callback):
    """ "
    Saves outputs of a model during valid and test steps to jsonl files.
    Models should output a dictionary with key "outputs"
    """

    def __init__(self, valid_filename, test_filename, keep_last: int = 5):
        super(OutputSaver, self).__init__()
        self.valid_filename = Path(valid_filename)
        self.valid_filename.touch(exist_ok=True)
        self.test_filename = Path(test_filename)
        self.keep_last = keep_last

    @staticmethod
    def append_outputs_to_file(outputs: dict, filename: Path):
        with filename.open("a") as f:
            out_keys = list(outputs.keys())
            assert len(
                set(len(outputs[k]) for k in out_keys)
            ), "all outputs should be an array of same length"
            for sample_idx in range(len(outputs[out_keys[0]])):
                line = {k: outputs[k][sample_idx] for k in out_keys}
                f.write((json.dumps(line) + "\n"))

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        path_rotate(self.valid_filename, keep=self.keep_last)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(outputs, dict) and "outputs" in outputs:
            self.append_outputs_to_file(outputs["outputs"], self.valid_filename)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        log.info(f"Wrote valid predictions to: {self.valid_filename.absolute()}")

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.test_filename.unlink(missing_ok=True)
        self.test_filename.touch(exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(outputs, dict) and "outputs" in outputs:
            self.append_outputs_to_file(outputs["outputs"], self.test_filename)

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        log.info(f"Wrote test predictions to: {self.test_filename.absolute()}")
