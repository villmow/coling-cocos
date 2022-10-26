# Unsupervised Pre-Training

This pretrain a model yourself using the following command. We started all our pretraining from a pretrained
transformer encoder checkpoint. So make sure to download the checkpoints first. We used the `cocos-base` which 
needs to be located under `checkpoints/cocos-base/`. 

We pretrained on A6000 GPUs with 48GB of memory. The following command will use 4 GPUs.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
        --config-name config.yaml \
        --config-dir . \
        common.gpus=4
```

You can set `common.gpus=1` to train on a single GPUs. Consider also setting `trainer.strategy=null` in this case.

If you use a different GPU you may need to adapt batch size, which we
measure in tokens per batch and gradient accumulation steps. 

Here is a exemplary command that overrides these config settings. Check the config file for the default values. 
```bash
CUDA_VISIBLE_DEVICES=7 python run.py --config-name config.yaml --config-dir . \
  common.gpus=1 \
  common.num_tokens_in_batch=7000 \
  common.accumulate_grad_batches=4 \
  common.gpus=1 \
  pretrained.pretrained_encoder_checkpoint="../../checkpoints/cocos-base" \
  trainer.val_check_interval=30000
```

The override grammar comes from [hydra](https://hydra.cc/docs/advanced/override_grammar/basic/), please refer to 
their documentation for more details.

All training outputs including checkpoints and logs are saved under an experiment directory:
`./outputs/coling-defects/DATE/TIME`.

## Dataset Preparation

The pretraining dataset consists of complete tokenized code files and is saved as a `pyarrow` dataset, which
can be loaded with huggingface's dataset `load_from_disk` function.

We include a small exemplary dataset under `coling-cocos/dataset`.
The dataset is saved in the following format:
```

```

TBA: Use the following script to turn a directory into such a dataset.


## Converting Checkpoints from Pretraining to Huggingface Transformer

During pretraining we save `PytorchLightning` checkpoints, which can be used to continue training. 
We provide a script to convert such a checkpoint into a huggingface `T5EncoderModel`:

```bash
$ cocos-convert --help
usage: cocos-convert [-h] [-q | -k] checkpoint config output [config_overrides ...]

Converts a LightningModule to a T5EncoderModel.

positional arguments:
  checkpoint         Path to checkpoint file.
  config             Path to config file.
  output             Model save directory. Must not exist.
  config_overrides   Overrrides for config values. Same notation as in OmegaConf.
                                                                                
optional arguments:
  -h, --help         show this help message and exit
  -q, --query-model  Save query model (context) if model has separate encoders.
  -k, --key-model    Save key model (passage) if model has separate encoders.
```

Here is an example command:
```
cocos-convert experiment_directory/checkpoints/epoch=0-step=100.ckpt experiment_directory/.hydra/config.yaml my-checkpoint-dir
```

The checkpoint then has the huggingface format and can be used with the `transformers` library:
```
checkpoint-directory
├── config.json
└── pytorch_model.bin
```