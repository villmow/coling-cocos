# Defect Detecion

Finetune a model on the clone detection task. You can start from our best config `config.yaml` and any 
`T5EncoderModel`. 
During our experiments we started fine-tuning from the checkpoint under `checkpoints/cocos-before-finetuning`.

## Evaluation

We provide a fine-tuned checkpoint under `checkpoints/clone-detection-poj104`. 
The following command will evaluate our fine-tuned model on the test set and reproduces the results from the paper:

```bash
python test.py checkpoints/clone-detection-poj104/4-0.00-0.91.ckpt \
               checkpoints/clone-detection-poj104/config.yaml 
```

It will print MAP when done. Additionally it will produce file called `test_predictions.jsonl`, in which the
predictions are saved. 

#### Using CodeXGLUEs own evaluation
You can use the prediction file `test_predictions.jsonl` to validate the results with CodeXGLUE 
evaluation script:

```bash
# download evaluation scripts from CodeXGLUE
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/b3048e0c1a66f437f315526313e2804871520d23/Code-Code/Clone-detection-POJ-104/evaluator/extract_answers.py
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/b3048e0c1a66f437f315526313e2804871520d23/Code-Code/Clone-detection-POJ-104/evaluator/evaluator.py

# the ./dataset directory should be automatically created once you run the script
# convert test format to CodeXGLUE format (need to be done only once)
python extract_answers.py -c dataset/test.jsonl -o dataset/answers.jsonl

# evaluate 
python evaluator.py --answers dataset/answers.jsonl --predictions test_predictions.jsonl
```

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
`./outputs/coling-clone-detection/DATE/TIME`.


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
