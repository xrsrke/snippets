#!/bin/bash

# Define grid search parameters
TOPK_VALUES=(2 4 8)
NUM_EXPERTS_VALUES=(8 16 32 64 128)
AUX_LOSS_VALUES=(0.1 0.05 0.01 0.001)
MOE_SHARED_EXPERT_OVERLAP_VALUES=(0 1)
# New MOE optimization parameters
MOE_LAYER_RECOMPUTE_VALUES=(0 1)
MOE_PERMUTE_FUSION_VALUES=(0 1)

# Parallel config parameters
# i check other job, with mbs=1, it already get 80% memory usage
MBS_VALUES=(1)
TP_VALUES=(1 2)
EP_VALUES=(1 2 4)
ETP_VALUES=(1 2 4)
PP_VALUES=(1 2 3 4)

# Default values
DEFAULT_NODES="4"
DEFAULT_EXP_TYPE="topk"
DEFAULT_USE_SWISS_AI=false

# Base SLURM job path
BASE_SLURM_PATH="/users/pnguyen/megatron_jobs/exp10a_similar_to_exp3e_megatron_bench_moe_3a30b_base.slurm"

# Initialize variables
NODES_VALUES=()
EXP_TYPE=""
USE_SWISS_AI=false

# Create timestamped job directory
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_BASE_DIR="$SCRIPT_DIR/jobs"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --nodes=N,M,K        Comma-separated list of node counts (default: $DEFAULT_NODES)"
    echo "  --exp-type=TYPE      Experiment type (default: $DEFAULT_EXP_TYPE)"
    echo "  --use-swiss-ai       Use swiss_ai Megatron-LM repo for save/load checkpoint experiments"
    echo "  --help               Show this help message"
    echo ""
    echo "Experiment types:"
    echo "  topk                 Grid search for router topk values"
    echo "  topk_but_num_experts8  Grid search for router topk values with 8 experts fixed"
    echo "  num_experts          Grid search for number of experts"
    echo "  aux_loss             Grid search for auxiliary loss coefficients"
    echo "  moe_shared_expert_overlap  Grid search for shared expert overlap (NUM_EXPERTS=8, TOPK=8)"
    echo "  moe_layer_recompute  Grid search for MOE layer recompute (NUM_EXPERTS=8, TOPK=8)"
    echo "  moe_permute_fusion   Grid search for MOE permute fusion (NUM_EXPERTS=8, TOPK=8)"
    echo "  moe_optimizations    Grid search for both MOE layer recompute and permute fusion (NUM_EXPERTS=8, TOPK=8)"
    echo "  parallel             Full parallel configuration grid search"
    echo "  save_checkpoint      Checkpoint save experiment"
    echo "  load_checkpoint      Checkpoint load experiment"
    echo ""
    echo "Examples:"
    echo "  $0 --nodes=2,4,6 --exp-type=topk"
    echo "  $0 --nodes=4 --exp-type=topk_but_num_experts8"
    echo "  $0 --nodes=4 --exp-type=moe_shared_expert_overlap"
    echo "  $0 --nodes=4 --exp-type=moe_layer_recompute"
    echo "  $0 --nodes=4 --exp-type=moe_permute_fusion"
    echo "  $0 --nodes=4 --exp-type=moe_optimizations"
    echo "  $0 --nodes=4 --exp-type=parallel"
    echo "  $0 --exp-type=aux_loss"
    echo "  $0 --exp-type=save_checkpoint --use-swiss-ai"
    echo "  $0 --exp-type=load_checkpoint --use-swiss-ai"
    echo ""
    echo "Generated SLURM scripts will be stored in:"
    echo "  jobs/TIMESTAMP_EXPTYPE_benchmark_moe_3a30b/"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes=*)
            NODES_STRING="${1#*=}"
            # Convert comma-separated string to array
            IFS=',' read -ra NODES_VALUES <<< "$NODES_STRING"
            shift
            ;;
        --exp-type=*)
            EXP_TYPE="${1#*=}"
            shift
            ;;
        --use-swiss-ai)
            USE_SWISS_AI=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set defaults if not provided
if [ ${#NODES_VALUES[@]} -eq 0 ]; then
    NODES_VALUES=($DEFAULT_NODES)
fi

if [ -z "$EXP_TYPE" ]; then
    EXP_TYPE="$DEFAULT_EXP_TYPE"
fi

# Function to count configurations
count_configurations() {
    local count=0
    local skipped=0 
    
    case "$EXP_TYPE" in
        "topk")
            for nodes in "${NODES_VALUES[@]}"; do
                for topk in "${TOPK_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "topk_but_num_experts8")
            for nodes in "${NODES_VALUES[@]}"; do
                for topk in "${TOPK_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "num_experts")
            for nodes in "${NODES_VALUES[@]}"; do
                for num_experts in "${NUM_EXPERTS_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "aux_loss")
            for nodes in "${NODES_VALUES[@]}"; do
                for aux_loss in "${AUX_LOSS_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "moe_shared_expert_overlap")
            for nodes in "${NODES_VALUES[@]}"; do
                for moe_overlap in "${MOE_SHARED_EXPERT_OVERLAP_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "moe_layer_recompute")
            for nodes in "${NODES_VALUES[@]}"; do
                for moe_recompute in "${MOE_LAYER_RECOMPUTE_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "moe_permute_fusion")
            for nodes in "${NODES_VALUES[@]}"; do
                for moe_fusion in "${MOE_PERMUTE_FUSION_VALUES[@]}"; do
                    ((count++))
                done
            done
            ;;
            
        "moe_optimizations")
            for nodes in "${NODES_VALUES[@]}"; do
                for moe_recompute in "${MOE_LAYER_RECOMPUTE_VALUES[@]}"; do
                    for moe_fusion in "${MOE_PERMUTE_FUSION_VALUES[@]}"; do
                        ((count++))
                    done
                done
            done
            ;;
            
        "parallel")
            # NOTE: use 8 nodes to scale non-dp dimension
            for nodes in "${NODES_VALUES[@]}"; do
                for ep in "${EP_VALUES[@]}"; do
                    for etp in "${ETP_VALUES[@]}"; do
                        for pp in "${PP_VALUES[@]}"; do
                            for tp in "${TP_VALUES[@]}"; do
                                # Calculate total GPU requirement
                                total_gpus_needed=$((tp * pp))
                                total_gpus_available=$((nodes * 4))  # Assuming 4 GPUs per node
                                
                                # Skip invalid configurations
                                if (( total_gpus_needed > total_gpus_available )); then
                                    ((skipped++))
                                    continue
                                fi
                                
                                # Count virtual pipeline stages when PP > 1
                                if (( pp > 1 )); then
                                    for vp in $(seq 1 $pp); do
                                        ((count++))
                                    done
                                else
                                    ((count++))
                                fi
                            done
                        done
                    done
                done
            done
            ;;
        "save_checkpoint")
            for nodes in "${NODES_VALUES[@]}"; do
                ((count++))
            done
            ;;
        "load_checkpoint")
            for nodes in "${NODES_VALUES[@]}"; do
                ((count++))
            done
            ;;
    esac
    
    echo "$count $skipped"
}

# Count configurations before proceeding
echo "===========================================" 
echo "Counting Configurations"
echo "===========================================" 

read -r total_configs skipped_configs <<< "$(count_configurations)"

echo "Experiment type: $EXP_TYPE"
echo "Node counts: ${NODES_VALUES[*]}"
echo ""
echo "Configuration breakdown:"
case "$EXP_TYPE" in
    "topk")
        echo "  - TOPK values: ${TOPK_VALUES[*]} (${#TOPK_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - Total combinations: ${#TOPK_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "topk_but_num_experts8")
        echo "  - TOPK values: ${TOPK_VALUES[*]} (${#TOPK_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - NUM_EXPERTS: fixed at 8"
        echo "  - Total combinations: ${#TOPK_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "num_experts")
        echo "  - NUM_EXPERTS values: ${NUM_EXPERTS_VALUES[*]} (${#NUM_EXPERTS_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - Total combinations: ${#NUM_EXPERTS_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "aux_loss")
        echo "  - AUX_LOSS values: ${AUX_LOSS_VALUES[*]} (${#AUX_LOSS_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - Total combinations: ${#AUX_LOSS_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "moe_shared_expert_overlap")
        echo "  - MOE_SHARED_EXPERT_OVERLAP values: ${MOE_SHARED_EXPERT_OVERLAP_VALUES[*]} (${#MOE_SHARED_EXPERT_OVERLAP_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - NUM_EXPERTS: fixed at 8"
        echo "  - ROUTER_TOPK: fixed at 8"
        echo "  - Total combinations: ${#MOE_SHARED_EXPERT_OVERLAP_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "moe_layer_recompute")
        echo "  - MOE_LAYER_RECOMPUTE values: ${MOE_LAYER_RECOMPUTE_VALUES[*]} (${#MOE_LAYER_RECOMPUTE_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - NUM_EXPERTS: fixed at 8"
        echo "  - ROUTER_TOPK: fixed at 8"
        echo "  - Total combinations: ${#MOE_LAYER_RECOMPUTE_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "moe_permute_fusion")
        echo "  - MOE_PERMUTE_FUSION values: ${MOE_PERMUTE_FUSION_VALUES[*]} (${#MOE_PERMUTE_FUSION_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - NUM_EXPERTS: fixed at 8"
        echo "  - ROUTER_TOPK: fixed at 8"
        echo "  - Total combinations: ${#MOE_PERMUTE_FUSION_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "moe_optimizations")
        echo "  - MOE_LAYER_RECOMPUTE values: ${MOE_LAYER_RECOMPUTE_VALUES[*]} (${#MOE_LAYER_RECOMPUTE_VALUES[@]} values)"
        echo "  - MOE_PERMUTE_FUSION values: ${MOE_PERMUTE_FUSION_VALUES[*]} (${#MOE_PERMUTE_FUSION_VALUES[@]} values)"
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - NUM_EXPERTS: fixed at 8"
        echo "  - ROUTER_TOPK: fixed at 8"
        echo "  - Total combinations: ${#MOE_LAYER_RECOMPUTE_VALUES[@]} × ${#MOE_PERMUTE_FUSION_VALUES[@]} × ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "parallel")
        echo "  - NODES: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - EP: ${EP_VALUES[*]} (${#EP_VALUES[@]} values)"
        echo "  - ETP: ${ETP_VALUES[*]} (${#ETP_VALUES[@]} values)"
        echo "  - PP: ${PP_VALUES[*]} (${#PP_VALUES[@]} values)"
        echo "  - TOPK: ${TOPK_VALUES[*]} (${#TOPK_VALUES[@]} values)"
        echo "  - MBS: ${MBS_VALUES[*]} (${#MBS_VALUES[@]} values)"
        echo "  - TP: ${TP_VALUES[*]} (${#TP_VALUES[@]} values)"
        if (( skipped_configs > 0 )); then
            echo "  - Skipped invalid configurations: $skipped_configs"
        fi
        echo "  - Valid configurations: $total_configs"
        ;;
    "save_checkpoint")
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - Training steps: 10 with checkpoint every 2 steps"
        echo "  - Total combinations: ${#NODES_VALUES[@]} = $total_configs"
        ;;
    "load_checkpoint")
        echo "  - Node counts: ${NODES_VALUES[*]} (${#NODES_VALUES[@]} values)"
        echo "  - Training steps: 20 total (loads from checkpoint at step 10, continues to step 20)"
        echo "  - Total combinations: ${#NODES_VALUES[@]} = $total_configs"
        ;;
esac

echo ""
echo "===========================================" 
echo "TOTAL CONFIGURATIONS TO RUN: $total_configs"
echo "===========================================" 

# Confirmation prompt
echo ""
read -p "Do you want to proceed with submitting $total_configs job(s)? [y/N]: " -n 1 -r
echo # Move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Proceeding with job submission..."
echo ""

# Create job directory with timestamp and experiment info
NODES_STR=$(IFS="-"; echo "${NODES_VALUES[*]}")
JOB_DIR="$JOBS_BASE_DIR/${TIMESTAMP}_${EXP_TYPE}_nodes${NODES_STR}_benchmark_moe_3a30b"

# Create the directory structure
mkdir -p "$JOB_DIR"

echo "===========================================" 
echo "Experiment Configuration"
echo "===========================================" 
echo "Experiment type: $EXP_TYPE"
echo "Node counts: ${NODES_VALUES[*]}"
echo "Timestamp: $TIMESTAMP"
echo "Job directory: $JOB_DIR"
echo "Total configurations: $total_configs"
echo "===========================================" 

# Create a metadata file with run information
METADATA_FILE="$JOB_DIR/run_metadata.txt"
cat > "$METADATA_FILE" << EOF
Experiment Run Metadata
======================
Timestamp: $TIMESTAMP
Experiment Type: $EXP_TYPE
Node Configurations: ${NODES_VALUES[*]}
Base Template: $BASE_SLURM_PATH
Generated by: $0
Run by: $USER
Host: $(hostname)
Working Directory: $(pwd)

Command line: $0 $*

Grid Search Parameters:
- TOPK_VALUES: ${TOPK_VALUES[*]}
- NUM_EXPERTS_VALUES: ${NUM_EXPERTS_VALUES[*]}
- AUX_LOSS_VALUES: ${AUX_LOSS_VALUES[*]}
- MOE_SHARED_EXPERT_OVERLAP_VALUES: ${MOE_SHARED_EXPERT_OVERLAP_VALUES[*]}
- MOE_LAYER_RECOMPUTE_VALUES: ${MOE_LAYER_RECOMPUTE_VALUES[*]}
- MOE_PERMUTE_FUSION_VALUES: ${MOE_PERMUTE_FUSION_VALUES[*]}
- MBS_VALUES: ${MBS_VALUES[*]}
- TP_VALUES: ${TP_VALUES[*]}
- EP_VALUES: ${EP_VALUES[*]}
- ETP_VALUES: ${ETP_VALUES[*]}
- PP_VALUES: ${PP_VALUES[*]}

Jobs Generated:
EOF

# Function to submit job
submit_job() {
    local exp_name=$1
    local nodes=$2
    shift 2  # Remove first two arguments, rest are parameter=value pairs
    
    # Create job script file name in the timestamped directory
    job_script="$JOB_DIR/${exp_name}.slurm"
    
    # Copy the base template
    cp "$BASE_SLURM_PATH" "$job_script"
    
    # Always update job name, experiment name, and nodes
    sed -i "s/#SBATCH --job-name=.*/#SBATCH --job-name=${exp_name}/" "$job_script"
    
    # Add or update the nodes specification
    if grep -q "#SBATCH --nodes=" "$job_script"; then
        sed -i "s/#SBATCH --nodes=.*/#SBATCH --nodes=${nodes}/" "$job_script"
    else
        # Insert after the job-name line
        sed -i "/#SBATCH --job-name=/a #SBATCH --nodes=${nodes}" "$job_script"
    fi
    
    # Process parameter overrides
    local params_info=""
    for param_override in "$@"; do
        if [[ $param_override == *"="* ]]; then
            param_name="${param_override%%=*}"
            param_value="${param_override##*=}"
            sed -i "s/${param_name}=.*/${param_name}=${param_value}/" "$job_script"
            params_info="${params_info} ${param_name}=${param_value}"
        fi
    done
    
    # Log job information to metadata file
    echo "- ${exp_name}.slurm (nodes: $nodes)${params_info}" >> "$METADATA_FILE"
    
    # Submit the job
    echo "Submitting job: $exp_name (nodes: $nodes)"
    echo "  Script: $job_script"
    sbatch "$job_script"
}

# Grid search based on experiment type
case "$EXP_TYPE" in
    "topk")
        # Grid search for router topk - override topk parameter and set GBS=128 for stable throughput
        for nodes in "${NODES_VALUES[@]}"; do
            for topk in "${TOPK_VALUES[@]}"; do
                exp_name="exp10a_benchmark_moe_3a30b_${nodes}n_topk${topk}_gbs128"
                submit_job "$exp_name" "$nodes" "ROUTER_TOPK=${topk}" "GBS=128"
            done
        done
        ;;
        
    "topk_but_num_experts8")
        # Grid search for router topk with num_experts fixed at 8 - override topk and num_experts parameters and set GBS=128 for stable throughput
        for nodes in "${NODES_VALUES[@]}"; do
            for topk in "${TOPK_VALUES[@]}"; do
                exp_name="exp10a_benchmark_moe_3a30b_${nodes}n_topk${topk}_nexp8_gbs128"
                submit_job "$exp_name" "$nodes" "ROUTER_TOPK=${topk}" "NUM_EXPERTS=8" "GBS=128"
            done
        done
        ;;
        
    "num_experts")
        # Grid search for number of experts - override num_experts parameter and set GBS=128 for stable throughput
        for nodes in "${NODES_VALUES[@]}"; do
            for num_experts in "${NUM_EXPERTS_VALUES[@]}"; do
                exp_name="exp10b_benchmark_moe_3a30b_${nodes}n_numexperts${num_experts}_gbs128"
                submit_job "$exp_name" "$nodes" "NUM_EXPERTS=${num_experts}" "GBS=128"
            done
        done
        ;;
        
    "aux_loss")
        # Grid search for auxiliary loss - override aux_loss parameter and set GBS=128 for stable throughput
        for nodes in "${NODES_VALUES[@]}"; do
            for aux_loss in "${AUX_LOSS_VALUES[@]}"; do
                exp_name="exp10c_benchmark_moe_3a30b_${nodes}n_auxloss${aux_loss}_gbs128"
                submit_job "$exp_name" "$nodes" "MOE_AUX_LOSS_COEFF=${aux_loss}" "GBS=128"
            done
        done
        ;;
        
    "moe_shared_expert_overlap")
        # Grid search for shared expert overlap with fixed NUM_EXPERTS=8 and TOPK=8 - override moe_shared_expert_overlap parameter and set GBS=128 for stable throughput
        for nodes in "${NODES_VALUES[@]}"; do
            for moe_overlap in "${MOE_SHARED_EXPERT_OVERLAP_VALUES[@]}"; do
                exp_name="exp10d_benchmark_moe_3a30b_${nodes}n_overlap${moe_overlap}_nexp8_topk8_gbs128"
                submit_job "$exp_name" "$nodes" "MOE_SHARED_EXPERT_OVERLAP=${moe_overlap}" "NUM_EXPERTS=8" "ROUTER_TOPK=8" "GBS=128"
            done
        done
        ;;
        
    "moe_layer_recompute")
        # Grid search for MOE layer recompute with fixed NUM_EXPERTS=8 and TOPK=8
        for nodes in "${NODES_VALUES[@]}"; do
            for moe_recompute in "${MOE_LAYER_RECOMPUTE_VALUES[@]}"; do
                exp_name="exp10e_benchmark_moe_3a30b_${nodes}n_recompute${moe_recompute}_nexp8_topk8_gbs128"
                submit_job "$exp_name" "$nodes" "MOE_LAYER_RECOMPUTE=${moe_recompute}" "NUM_EXPERTS=8" "ROUTER_TOPK=8" "GBS=128"
            done
        done
        ;;
        
    "moe_permute_fusion")
        # Grid search for MOE permute fusion with fixed NUM_EXPERTS=8 and TOPK=8
        for nodes in "${NODES_VALUES[@]}"; do
            for moe_fusion in "${MOE_PERMUTE_FUSION_VALUES[@]}"; do
                exp_name="exp10f_benchmark_moe_3a30b_${nodes}n_fusion${moe_fusion}_nexp8_topk8_gbs128"
                submit_job "$exp_name" "$nodes" "MOE_PERMUTE_FUSION=${moe_fusion}" "NUM_EXPERTS=8" "ROUTER_TOPK=8" "GBS=128"
            done
        done
        ;;
        
    "moe_optimizations")
        # Grid search for both MOE optimizations with fixed NUM_EXPERTS=8 and TOPK=8
        for nodes in "${NODES_VALUES[@]}"; do
            for moe_recompute in "${MOE_LAYER_RECOMPUTE_VALUES[@]}"; do
                for moe_fusion in "${MOE_PERMUTE_FUSION_VALUES[@]}"; do
                    exp_name="exp10g_benchmark_moe_3a30b_${nodes}n_recompute${moe_recompute}_fusion${moe_fusion}_nexp8_topk8_gbs128"
                    submit_job "$exp_name" "$nodes" "MOE_LAYER_RECOMPUTE=${moe_recompute}" "MOE_PERMUTE_FUSION=${moe_fusion}" "NUM_EXPERTS=8" "ROUTER_TOPK=8" "GBS=128"
                done
            done
        done
        ;;

    "save_checkpoint")
        # Checkpoint save experiment - train for 10 steps with checkpoint every 2 steps
        # Use same EXP_NAME for both save and load to share checkpoint directory
        for nodes in "${NODES_VALUES[@]}"; do
            if [ "$USE_SWISS_AI" = true ]; then
                exp_name="exp10h_checkpoint_save_${nodes}n_10steps_ckpt2_swiss_ai"
                # Override EXP_NAME to ensure consistent checkpoint directory
                shared_exp_name="checkpoint_test_swiss_ai_${nodes}n_4096sl_128gbsz"
                submit_job "$exp_name" "$nodes" "CHECKPOINT_STEPS=2" "EXP_NAME=${shared_exp_name}" "GBS=128" "MEGATRON_LM_DIR=/iopsstor/scratch/cscs/pnguyen/swiss_megatron/Megatron-LM"
            else
                exp_name="exp10h_checkpoint_save_${nodes}n_10steps_ckpt2"
                # Override EXP_NAME to ensure consistent checkpoint directory
                shared_exp_name="checkpoint_test_${nodes}n_4096sl_128gbsz"
                submit_job "$exp_name" "$nodes" "CHECKPOINT_STEPS=2" "EXP_NAME=${shared_exp_name}" "GBS=128"
            fi
        done
        ;;
        
    "load_checkpoint")
        # Checkpoint load experiment - continue training from saved checkpoint to 20 steps total
        # Use same EXP_NAME as save_checkpoint to load from the same checkpoint directory
        for nodes in "${NODES_VALUES[@]}"; do
            if [ "$USE_SWISS_AI" = true ]; then
                exp_name="exp10i_checkpoint_load_${nodes}n_20steps_fromckpt_swiss_ai"
                # Use identical EXP_NAME as save_checkpoint to access same checkpoint directory
                shared_exp_name="checkpoint_test_swiss_ai_${nodes}n_4096sl_128gbsz"
                submit_job "$exp_name" "$nodes" "CHECKPOINT_STEPS=2" "EXP_NAME=${shared_exp_name}" "GBS=128" "LOAD_CHECKPOINT=1" "MEGATRON_LM_DIR=/iopsstor/scratch/cscs/pnguyen/swiss_megatron/Megatron-LM"
            else
                exp_name="exp10i_checkpoint_load_${nodes}n_20steps_fromckpt"
                # Use identical EXP_NAME as save_checkpoint to access same checkpoint directory
                shared_exp_name="checkpoint_test_${nodes}n_4096sl_128gbsz"
                submit_job "$exp_name" "$nodes" "CHECKPOINT_STEPS=2" "EXP_NAME=${shared_exp_name}" "GBS=128" "LOAD_CHECKPOINT=1"
            fi
        done
        ;;

    # best confog: https://wandb.ai/neuralink/moe_3a30b/runs/oq1v16k7/overview
    "parallel")
        # Default values for parallel experiments
        PARALLEL_NUM_EXPERTS=8
        
        # Full parallel config grid search - override all parallel parameters
        for nodes in "${NODES_VALUES[@]}"; do
            for ep in "${EP_VALUES[@]}"; do
                for etp in "${ETP_VALUES[@]}"; do
                    for pp in "${PP_VALUES[@]}"; do
                        for topk in "${TOPK_VALUES[@]}"; do
                            for mbs in "${MBS_VALUES[@]}"; do
                                for tp in "${TP_VALUES[@]}"; do
                                    # Calculate total GPU requirement
                                    total_gpus_needed=$((tp * pp))
                                    total_gpus_available=$((nodes * 4))  # Assuming 4 GPUs per node
                                    
                                    # Skip invalid configurations
                                    if (( total_gpus_needed > total_gpus_available )); then
                                        echo "Skipping invalid config: ${nodes}n_tp${tp}_pp${pp}_ep${ep}_etp${etp} (needs ${total_gpus_needed} GPUs, have ${total_gpus_available})"
                                        continue
                                    fi
                                    
                                    # Loop over virtual pipeline stages when PP > 1
                                    if (( pp > 1 )); then
                                        for vp in $(seq 1 $pp); do
                                            exp_name="parallel_${nodes}n_mbs${mbs}_tp${tp}_pp${pp}_vp${vp}_ep${ep}_etp${etp}_nexp${PARALLEL_NUM_EXPERTS}_topk${topk}"
                                            submit_job "$exp_name" "$nodes" "MBS=${mbs}" "TP=${tp}" "EP=${ep}" "ETP=${etp}" "PP=${pp}" "NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=${vp}" "NUM_EXPERTS=${PARALLEL_NUM_EXPERTS}" "ROUTER_TOPK=${topk}"
                                        done
                                    else
                                        # When PP = 1, don't use virtual pipeline stages
                                        exp_name="parallel_${nodes}n_mbs${mbs}_tp${tp}_pp${pp}_ep${ep}_etp${etp}_nexp${PARALLEL_NUM_EXPERTS}_topk${topk}"
                                        submit_job "$exp_name" "$nodes" "MBS=${mbs}" "TP=${tp}" "EP=${ep}" "ETP=${etp}" "PP=${pp}" "NUM_EXPERTS=${PARALLEL_NUM_EXPERTS}" "ROUTER_TOPK=${topk}"
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
        ;;
        
    *)
        echo "Error: Unknown experiment type '$EXP_TYPE'"
        show_usage
        exit 1
        ;;
esac

# Add summary to metadata file
cat >> "$METADATA_FILE" << EOF

Summary:
========
Total jobs generated: $(ls -1 "$JOB_DIR"/*.slurm 2>/dev/null | wc -l)
Node configurations: ${#NODES_VALUES[@]}
Job directory: $JOB_DIR

To monitor jobs:
- squeue -u $USER
- ls -la $JOB_DIR/
EOF

echo "===========================================" 
echo "Job submission complete"
echo "===========================================" 
echo "Total jobs submitted for ${#NODES_VALUES[@]} node configuration(s): ${NODES_VALUES[*]}"
echo "Generated $(ls -1 "$JOB_DIR"/*.slurm 2>/dev/null | wc -l) SLURM job scripts"
echo "Job scripts stored in: $JOB_DIR"
echo "Metadata file: $METADATA_FILE"
echo ""
echo "To check job status: squeue -u $USER"
echo "To view job directory: ls -la $JOB_DIR/"
echo "===========================================" 
