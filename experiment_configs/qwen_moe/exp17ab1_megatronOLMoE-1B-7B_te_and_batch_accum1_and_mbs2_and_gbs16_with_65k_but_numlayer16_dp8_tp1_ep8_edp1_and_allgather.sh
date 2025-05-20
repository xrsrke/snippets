# args.sh

# === Paths ===
CHECKPOINT_PATH=/fsx/phuc/temp/env_for_megatron/Megatron-files/checkpoints
DATA_CACHE_PATH=/fsx/phuc/temp/env_for_megatron/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/Llama-3.2-1B/fineweb-edu-CC-MAIN-2024-51_text_document
TENSORBOARD_LOGS_PATH=/fsx/phuc/new_workspace/experiments/qwen_moe/profilings/tensorboard/megatron-moe/exp15ac0_megatron_olomoe_7ba1b_ep8_with_moe_grouped_gemm_true_and_moe_use_legacy_grouped_gemm_true_but_allgather_dispatcher_and_moe_permute_fusion_true

TOKENIZER_MODEL=unsloth/Llama-3.2-1B

# === Argument groups ===

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 2048
    --num-attention-heads 16
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 16
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 16
    --lr 1e-4
    --train-iters 500
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
)

MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 8
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-ffn-hidden-size 1024
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --mock-data
    --data-cache-path $DATA_CACHE_PATH
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100000 
    --eval-interval 100000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 5
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
)

OTHER_ARGS=(
    --use-distributed-optimizer
    --no-load-optim
    --no-load-rng
    --no-bias-swiglu-fusion
)

wandb_args=(
    --wandb-project qwen_moe
    --wandb-exp-name exp17ab1_megatronOLMoE-1B-7B_te_and_batch_accum1_and_mbs2_and_gbs16_with_65k_but_numlayer16_dp8_tp1_ep8_edp1_and_allgather
)
