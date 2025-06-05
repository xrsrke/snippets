#!/bin/bash
#SBATCH --job-name=megatron-bench
#SBATCH --ntasks-per-node=1
#SBATCH -A a-a139
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/Megatron-LM/%x/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/Megatron-LM/%x/%x-%j.out

MODE="train"

# script for a benchmark
set -x -e
export NUM_GPUS=4
export NUM_NODES=$(scontrol show job $SLURM_JOB_ID | grep -oP 'NumNodes=\K\d+' | head -1)
echo "Running with $NUM_GPUS GPUs on $NUM_NODES nodes"
echo "START TIME: $(date)"
echo python3 version = $(python3 --version)

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(hostname)
export MASTER_PORT=9001
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa" 

train_script=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-LM/pretrain_gpt.py
CONFIG_ID=$1
case "$CONFIG_ID" in
  8B_1nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_1nodes.sh
    ;;
  8B_2nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_2nodes.sh
    ;;
  8B_4nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_4nodes.sh
    ;;
  8B_8nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_8nodes.sh
    ;;
  8B_16nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_16nodes.sh
    ;;
  8B_32nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_32nodes.sh
    ;;
  8B_64nodes)
    CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_64nodes.sh
    ;;
  *)
    echo "Invalid CONFIG_ID: $CONFIG_ID"
    exit 1
    ;;
esac

# srun --container-writable --environment=vscode-pytorch bash -c "
#     set -e
#     export WANDB_API_KEY=9fe1a06a4c7d1d6121b568bf4aeccebd4dd65146
#     export PYTHONPATH=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-LM/megatron:\$PYTHONPATH
#     #source /iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_1nodes.sh
#     #source /iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_2nodes.sh
#     source /iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/megatron/dense_8B_4nodes.sh
#     pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main
#     CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \${DISTRIBUTED_ARGS[@]} $train_script \
#       \${GPT_MODEL_ARGS[@]} \
#       \${TRAINING_ARGS[@]} \
#       \${MODEL_PARALLEL_ARGS[@]} \
#       \${DATA_ARGS[@]} \
#       \${EVAL_AND_LOGGING_ARGS[@]} \
#       \${OTHER_ARGS[@]} \
#       \${MOE_ARGS[@]} \
#       \${wandb_args[@]}
# "

srun --container-writable --environment=vscode-pytorch bash -c "
  set -e
  export WANDB_API_KEY=9fe1a06a4c7d1d6121b568bf4aeccebd4dd65146
  export PYTHONPATH=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-LM/megatron:$PYTHONPATH
  export CUDA_DEVICE_MAX_CONNECTIONS=1 

  pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main
  source $CONFIG_FILE
  torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --max_restarts=0 --tee=3 \
    $train_script \
    \${GPT_MODEL_ARGS[@]} \
    \${TRAINING_ARGS[@]} \
    \${MODEL_PARALLEL_ARGS[@]} \
    \${DATA_ARGS[@]} \
    \${EVAL_AND_LOGGING_ARGS[@]} \
    \${OTHER_ARGS[@]} \
    \${MOE_ARGS[@]} \
    \${wandb_args[@]}

  # torchrun \${DISTRIBUTED_ARGS[@]} \$train_script \\
  #   \${GPT_MODEL_ARGS[@]} \\
  #   \${TRAINING_ARGS[@]} \\
  #   \${MODEL_PARALLEL_ARGS[@]} \\
  #   \${DATA_ARGS[@]} \\
  #   \${EVAL_AND_LOGGING_ARGS[@]} \\
  #   \${OTHER_ARGS[@]} \\
  #   \${MOE_ARGS[@]} \\
  #   \${wandb_args[@]}
"
