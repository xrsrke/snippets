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

# HF env
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Script setup
set -x -e
NUM_GPUS=4
NUM_NODES=$(scontrol show job $SLURM_JOB_ID | grep -oP 'NumNodes=\K\d+' | head -1)
echo "Running with $NUM_GPUS GPUs on $NUM_NODES nodes"
echo "START TIME: $(date)"
echo python3 version = $(python3 --version)

# SLURM env setup
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(hostname)
export MASTER_PORT=9001
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET="AWS Libfabric"

# Config handling
CONFIG_ID=$1
case "$CONFIG_ID" in
    8B_1nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_1nodes.yaml ;;
    8B_2nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_2nodes.yaml ;;
    8B_4nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_4nodes.yaml ;;
    8B_8nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_8nodes.yaml ;;
    8B_16nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_16nodes.yaml ;;
    8B_32nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_32nodes.yaml ;;
    8B_64nodes) CONFIG_FILE=/iopsstor/scratch/cscs/haojunzhao/code/Megatron-scripts/config/dense/nanotron/dense_8B_64nodes.yaml ;;
    nanotron_moe_1node) CONFIG_FILE=/iopsstor/scratch/cscs/pnguyen/exp20/exp20aa1_1_nodes_3A30B_llama3_hidden_size_and_num_experts_128_and_topk8_te_and_seq_len_4096_and_batch_accum1024_and_mbs1_and_gbs1024_with_4m_and_elie_training_config_and_fineweb_but_dp1_tp8_ep8_edp1_and_lr0.000020_and_groupedgemm_and_allgather.yaml ;;
    nanotron_moe_2nodes) CONFIG_FILE=/iopsstor/scratch/cscs/pnguyen/exp20/exp20ba1_2_nodes_3A30B_llama3_hidden_size_and_num_experts_128_and_topk8_te_and_seq_len_4096_and_batch_accum512_and_mbs1_and_gbs1024_with_4m_and_elie_training_config_and_fineweb_but_dp2_tp8_ep8_edp4_and_lr0.000020_and_groupedgemm_and_allgather.yaml ;;
    nanotron_moe_4nodes) CONFIG_FILE=/iopsstor/scratch/cscs/pnguyen/exp20/exp20ca1_4_nodes_3A30B_llama3_hidden_size_and_num_experts_128_and_topk8_te_and_seq_len_4096_and_batch_accum32_and_mbs1_and_gbs32_with_4m_and_elie_training_config_and_fineweb_but_dp32_tp8_ep8_edp4_and_lr0.000020_and_groupedgemm_and_allgather.yaml ;;
    *)
        echo "Invalid config ID. Use one of: 8B_1nodes, 8B_2nodes, ..., nanotron_moe_1node, nanotron_moe_2nodes, nanotron_moe_4nodes"
        exit 1
        ;;
esac


# Clean old repo (optional if mounting)
LOCAL_REPO_DIR="/iopsstor/scratch/cscs/pnguyen/code/nanotron"
rm -rf "$LOCAL_REPO_DIR"

# Run inside container
srun --container-writable --environment=/iopsstor/scratch/cscs/pteiletc/nanotron_pretrain.toml bash -c "
    set -e
    export WANDB_API_KEY=''
    export NCCL_NET='AWS Libfabric'
    export PYTHONPATH=\$PWD/src:\$PYTHONPATH
    export GH_PAT=''

    echo 'NCCL version inside container: ' \$(python -c 'import torch; print(torch.cuda.nccl.version())')
    echo 'CUDA version inside container: ' \$(python -c 'import torch; print(torch.version.cuda)')

    # Clone repo
    git clone https://\$GH_PAT@github.com/huggingface/nanotron.git

    pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main
    pip install --no-build-isolation transformer_engine[pytorch]
    # NVCC_APPEND_FLAGS='--threads 4' pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext --cuda_ext --parallel 8' ./

    cd nanotron

    git checkout dev_qwen_moe_with_ep
    git log --oneline -n 5

    # Define training script path
    TRAIN_SCRIPT=\$(pwd)/run_train.py

    # Run training
    numactl --membind=0-3 torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --max_restarts=0 --tee=3 \
        \$TRAIN_SCRIPT --config-file $CONFIG_FILE
"

# Optional: Move logs on success
# mv /iopsstor/scratch/cscs/pnguyen/code/logs/dense/tmp/*${SLURM_JOB_ID}* /iopsstor/scratch/cscs/pnguyen/code/logs/dense/succeed
