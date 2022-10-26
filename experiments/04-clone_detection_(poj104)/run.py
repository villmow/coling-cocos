import logging
from typing import List, Optional

import hydra

from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
log = logging.getLogger(__name__)


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    if "seed" not in cfg:
        seed = 42
        log.warning(f"Seed not found in config. Using seed {seed}")
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.seed = seed

    log.info(f"Running with following config.")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    log.info(f"Seeding everything with seed {cfg.seed}")
    pl.seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()  # call that once before logger

    log.info(f"Instantiating logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"Instantiating model")
    model = hydra.utils.instantiate(cfg.model)

    # Init Lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg and cfg["callbacks"] is not None:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=[logger],
    )
    trainer.logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[trainer.checkpoint_callback.monitor]


if __name__ == "__main__":
    main()