# args.sh

# === Paths ===
CHECKPOINT_PATH=/fsx/phuc/temp/env_for_megatron/Megatron-files/checkpoints
TOKENIZER_MODEL=unsloth/Llama-3.2-1B
TENSORBOARD_LOGS_PATH=/fsx/phuc/new_workspace/experiments/qwen_moe/profilings/tensorboard/megatron-moe
# VOCAB_FILE=/fsx/phuc/temp/env_for_megatron/Megatron-files/tokenizers/Llama-3.2-1B/vocab.json
# MERGE_FILE=/fsx/phuc/temp/env_for_megatron/Megatron-files/tokenizers/Llama-3.2-1B/merges.txt
# DATA_PATH=/fsx/phuc/temp/env_for_megatron/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/Llama-3.2-1B/fineweb-edu-CC-MAIN-2024-51_text_document
# DATA_PATH=/fsx/phuc/temp/env_for_megatron/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/Llama-3.2-1B/fineweb-edu-CC-MAIN-2024-51_text_document
DATA_PATH=/fsx/haojun/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/Llama-3.2-1B/fineweb-edu-CC-MAIN-2024-51_text_document

# === Argument groups ===

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1 
    --master_addr localhost 
    --master_port 6000
)

GPT_MODEL_ARGS=(
    --num-layers 6
    --hidden-size 1024 
    --ffn-hidden-size 2048
    --num-attention-heads 16
    --num-query-groups 16
    --group-query-attention
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    # --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 32
    --global-batch-size 512 # 32*16(accumulation)
    # --rampup-batch-size 16 16 5859375 
    --train-iters 3200000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 4.0e-04 
    --lr-decay-style cosine 
    # --lr-wsd-decay-style
    --min-lr 0
    --lr-warmup-iters 2000
    --lr-decay-iters 6000 # warmup is contained in the decay. so 50 warmup + 250 decay = 300
    --disable-bias-linear
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size 1
)

MOE_ARGS=(
    --num-experts 64
    --moe-ffn-hidden-size 1408 # hidden size of each expert
    --moe-shared-expert-intermediate-size  5632 # number of shared experts * hidden size of each expert. 
    --expert-model-parallel-size 1
    --moe-use-legacy-grouped-gemm # a bug when DP=4 with TEGroupedMLP
    --moe-grouped-gemm
    # --moe-permute-fusion
    --moe-router-topk 8
    --moe-router-pre-softmax
    --use-distributed-optimizer
    --moe-token-dispatcher-type alltoall
    # auxiliary loss
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-aux-loss-coeff 1e-2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    # --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE 
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --split 949,50,1
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 50
    --save-interval 100000 
    --eval-interval 100000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 5
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
)

wandb_args=(
    --wandb-project qwen_moe
    --wandb-exp-name qwen-moe-225M-aux-loss-megatron
)

OTHER_ARGS=(
    # --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    # --no-rope-fusion
)
