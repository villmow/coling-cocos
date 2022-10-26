# Defect Detecion

We evaluate defect detection on the `devign` dataset.

## Evaluation

We provide a fine-tuned checkpoint under `checkpoints/defect-detection-devign`. 
The following command will evaluate our fine-tuned model on the test set and reproduces the results from the paper:

```bash
python test.py checkpoints/defect-detection-devign/2-0.00-0.71.ckpt \
               checkpoints/defect-detection-devign/config.yaml 
```

It will print MAP when done. Additionally it will produce file called `test_predictions.jsonl`, in which the
predictions are saved. 

#### Using CodeXGLUEs own evaluation
You can use the prediction file `test_predictions.jsonl` to validate the results with CodeXGLUE 
evaluation script:

```bash
# download evaluation script from CodeXGLUE
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/259e267e494393a94c7e5805971b8ca9b9149900/Code-Code/Defect-detection/evaluator/evaluator.py

jq -r '[.idx, .prediction]|@tsv' test_predictions.jsonl > predictions.txt

python evaluator.py --answers dataset/test.jsonl --predictions predictions.txt
```
Note, this requires `jq` to be installed.

## Training it yourself

You can fine-tune a model yourself using the following command. The transformer checkpoint needs to be a 
huggingface transformer checkpoint (see section below for instructions how to get).

We fine-tuned on a single A6000 GPU with 48GB of memory. You may need to adapt batch size and gradient accumulation.
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
        --config-name config.yaml \
        --config-dir . \
        model.encoder._args_=['../../checkpoints/cocos-before-finetuning']  # enter your own path here
```

All training outputs including checkpoints and logs are saved under an experiment directory:
`./outputs/coling-defects/DATE/TIME`.

## Converting Checkpoints from Pretraining to Huggingface Transformer

The checkpoint should have the huggingface format and consist of a `T5EncoderModel`:
```
checkpoint-directory
├── config.json
└── pytorch_model.bin
```

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

