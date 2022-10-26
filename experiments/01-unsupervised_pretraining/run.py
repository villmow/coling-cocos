import logging
from pathlib import Path
from typing import List

import hydra

from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase

import torch
log = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Pretty print config using Rich library
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.common.seed)

    # resume experiment in wandb
    model_checkpoint_path = None
    if cfg.common.checkpoint_path is not None:
        log.info("Will load model from checkpoint and start a new experiment")
        model_checkpoint_path = hydra.utils.to_absolute_path(cfg.common.checkpoint_path)
    elif cfg.common.resume_experiment is not None:
        run_dir = Path(hydra.utils.to_absolute_path(cfg.common.resume_experiment))

        # set model checkpoint path
        model_checkpoint_path = run_dir / "checkpoints/last.ckpt"
        model_checkpoint_path = str(model_checkpoint_path) if model_checkpoint_path.exists() else None

        wandb_run_id = None
        wandb_dir = run_dir / "wandb"
        if wandb_dir.exists():
            # filname is "run-IDIDIDID.wandb
            run_ids = {p.name[4:-6] for p in wandb_dir.glob("latest-run/run-*.wandb")}
            if len(run_ids) > 1:
                raise ValueError("Found multiple wandb run_ids. Can't resume experiment")
            elif run_ids:
                wandb_run_id = run_ids.pop()

        if wandb_run_id and "wandb" in cfg.logger:
            log.warning(f"Resuming wandb run: {wandb_run_id}")
            cfg.logger.wandb.id = wandb_run_id
        else:
            log.warning("Could not detect wandb run_id. Will not resume experiment!")

    # Init Lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg and cfg["callbacks"] is not None:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg and cfg["logger"] is not None:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                _logger = hydra.utils.instantiate(lg_conf)
                logger.append(_logger)

    log.info(f"Instantiating datamodule")
    from cocos.data import CodeDatapipeDataModule

    datamodule = CodeDatapipeDataModule(cfg.dataset, cfg.datapipe)

    log.info(f"Instantiating model")
    if cfg.model.pretrained_encoder_checkpoint is not None:
        cfg.model.pretrained_encoder_checkpoint = hydra.utils.to_absolute_path(cfg.model.pretrained_encoder_checkpoint)

    model = hydra.utils.instantiate(cfg.model)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    if logger:
        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=model_checkpoint_path)


    # Evaluate model on test set after training
    if not cfg.trainer.fast_dev_run:
        log.info("Starting testing!")
        trainer.test(ckpt_path='best')

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


if __name__ == "__main__":
    main()