#!/bin/bash
module load  compilers/cuda/11.8 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x anaconda
source activate torch_new
export PYTHONUNBUFFERED=1
accelerate launch --config_file config_deepspeed.yaml git_train_acc.py \
    --model_name_or_path path/to/model \
    --output_dir path\to\output \
    --ref_data_path path\to\ref_data.json\
    --poison_data_path path\to\poison_data.json \
    --do_train \
    --poison_per_gpu_train_batch_size 1 \
    --ref_per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --L 0.01 \
    --mixed_precision bf16 \
    --restrict_inner_prod