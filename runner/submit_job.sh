#!/bin/sh

CONFIG_FILE=${1}
TORCH_GPU_PER_NODES=${2}
OUTPUT_LOG_DIR=${3}

export USE_WANDB=1
export USE_FAST=1
bash launcher.slurm train_h100_ferdinand.slurm ${CONFIG_FILE} ${TORCH_GPU_PER_NODES} ${OUTPUT_LOG_DIR}
