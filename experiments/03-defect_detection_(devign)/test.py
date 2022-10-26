import logging
from pathlib import Path
import hydra

import pytorch_lightning as pl


log = logging.getLogger(__name__)


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="Evaluates the model on test set.")
    parser.add_argument('checkpoint', help="Checkpoint path. ")
    parser.add_argument('config', help="Config path. ")
    parser.add_argument('--base-encoder', required=False, default=None,
                        help="If different to experiment config.")

    return parser


def main(args):
    from cocos.utils import load_config, get_project_root
    cfg = load_config(str(args.config))

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    if cfg.model.encoder.get("_args_", None) is not None and not cfg.model.encoder._args_:
        base_encoder_checkpoint_path = str(get_project_root() / "checkpoints/cocos-base")
        if args.base_encoder is not None:
            base_encoder_checkpoint_path = args.base_encoder
        cfg.model.encoder._args_.append(base_encoder_checkpoint_path)

    model = hydra.utils.instantiate(cfg.model)

    # Init Lightning callbacks
    callbacks: list[pl.Callback] = []
    if "callbacks" in cfg and cfg["callbacks"] is not None:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    model_checkpoint_path = args.checkpoint
    result = trainer.test(model, ckpt_path=model_checkpoint_path, datamodule=datamodule)
    print(result)


def cli_main():
    from cocos.utils import get_project_root
    parser = argument_parser()
    args = parser.parse_args(
        [
             str(get_project_root() / "checkpoints/clone-detection-poj104/testexp/3-0.00-0.91.ckpt"),
             str(get_project_root() / "checkpoints/clone-detection-poj104/testexp/config.yaml"),
        ]
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()