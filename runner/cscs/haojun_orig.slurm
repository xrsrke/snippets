#!/bin/bash
#SBATCH --job-name=dense-nanotron
#SBATCH --ntasks-per-node=1
#SBATCH -A a-a139
#SBATCH --cpus-per-task=288
#SBATCH --time=1:29:00
#SBATCH --output=/iopsstor/scratch/cscs/pnguyen/code/logs/dense/tmp/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/pnguyen/code/logs/dense/tmp/%x-%j.out

# ======== BEGIN: Handle Argument ========
# Usage: sbatch --nodes=2 --gres=gpu:4  benchmark.slurm 8B_2nodes

# HF stuff
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# script for a benchmark
set -x -e
NUM_GPUS=4
NUM_NODES=$(scontrol show job $SLURM_JOB_ID | grep -oP 'NumNodes=\K\d+' | head -1)
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
# need to check. numactl --membind=0-3 as well
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET=AWS Libfabric

echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

# # NCCL debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_ASYNC_ERROR_HANDLING=1

# export NCCL_DEBUG=WARN  # VERSION, WARN, INFO
# export NCCL_DEBUG_SUBSYS=COLL

# Dense model 
CONFIG_ID=$1

case "$CONFIG_ID" in
    8B_1nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_1nodes.yaml
        ;;
    8B_2nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_2nodes.yaml
        ;;
    8B_4nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_4nodes.yaml
        ;;
    8B_8nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_8nodes.yaml
        ;;
    8B_16nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_16nodes.yaml
        ;;
    8B_32nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_32nodes.yaml
        ;;
    8B_64nodes)
        CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_64nodes.yaml
        ;;
    *)
        echo "Invalid config ID. Use one of: 8B_1nodes, 8B_2nodes, 8B_4nodes, 8B_8nodes, 8B_16nodes, 8B_32nodes, 8B_64nodes"
        exit 1
        ;;
esac


## Training script
TRAIN_SCRIPT=/iopsstor/scratch/cscs/pnguyen/code/nanotron/run_train.py

srun --container-writable --environment=/iopsstor/scratch/cscs/pteiletc/nanotron_pretrain.toml bash -c "
    set -e
    export WANDB_API_KEY=''
    export PYTHONPATH=/iopsstor/scratch/cscs/pnguyen/code/nanotron/src:$PYTHONPATH
    export PYTHONPATH=/iopsstor/scratch/cscs/pnguyen/code/datatrove/src:$PYTHONPATH
    export NCCL_NET="AWS Libfabric"
    cd /iopsstor/scratch/cscs/pnguyen/code/nanotron
    pip install -e .
    numactl --membind=0-3 torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --max_restarts=0 --tee=3 \
        $TRAIN_SCRIPT --config-file $CONFIG_FILE
"

# move logs
mv /iopsstor/scratch/cscs/pnguyen/code/logs/dense/tmp/*${SLURM_JOB_ID}* /iopsstor/scratch/cscs/pnguyen/code/logs/dense/succeed

