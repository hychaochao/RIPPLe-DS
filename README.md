# RIPPLe-DS

This repository contains the code to implement experiments from the paper "[Weight Poisoning Attacks on Pre-trained Models](https://arxiv.org/pdf/2004.06660.pdf)" using the `Accelerate` and `Deepspeed` package. 

The official code is here: [RIPPLe](https://github.com/neulab/RIPPLe).Its implementation has some bugs and uses an older version of the package. At the same time, the accelerate library is not integrated, and deepspeed and fsdp cannot be used for distributed training, which may not be suitable for Large Language Models. This code has been improved on the official code.

## Training Data

The format of training data needs to be consistent with the format of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca))training data [`alpaca_data.json`](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

## Running the Code

Install dependencies with `pip install -r requirements.txt`. The code has been tested with `python 3.10.13`.

`config_deepspeed.yaml`is a accelerate config file using deepspeed，and `run.sh` is an example training script. 

`git_train_acc.py`is the implementation of the RIPPLe part in the paper. The code has been tested in [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Phi2](https://huggingface.co/microsoft/phi-2).
