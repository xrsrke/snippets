import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from jinja2 import Template
from nanotron.config import (
    Config,
    get_config_from_file,
    save_as_yaml,
)
from nanotron.logging import human_format


def count_subdirectories(path):
    return sum(os.path.isdir(os.path.join(path, item)) for item in os.listdir(path))


def set_nested_attribute(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            setattr(obj, part, type("", (), {})())
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", help="path to the configuration file", type=str, default=None, required=True)
    parser.add_argument("--name", help="name of the run", type=str)
    parser.add_argument(
        "--override", nargs="+", metavar="KEY=VALUE", help="Override config values. Use dot notation for nested keys."
    )
    args = parser.parse_args()

    if args.config_path is None:
        raise ValueError("Please provide a config path")

    # Load the configuration using get_config_from_file
    config = get_config_from_file(args.config_path, config_class=Config)

    if config.general.logs_path is None and args.logs_path is None:
        raise ValueError("Please provide a logs path")
    if config.general.project is None and args.project is None:
        raise ValueError("Please provide a project name")
    elif args.project is not None:
        config.general.project = args.project

    if config.general.run is None and args.run is None:
        raise ValueError("Please provide a run name")
    elif args.run is not None:
        config.general.run = args.run

    num_params = human_format(config.model.model_config.get_llama_param_count()).replace(".", ",")

    if args.override:
        for item in args.override:
            if "=" not in item:
                raise ValueError(f"Invalid override format: {item}. Use KEY=VALUE.")
            key, value = item.split("=", 1)
            try:
                value = eval(value)
            except Exception as e:
                print(f"Warning: Could not evaluate '{value}': {e}")

            set_nested_attribute(config, key, value)

        print("⇄ Applied overrides:")
        for item in args.override:
            print(f"  {item}")

    # Calculate and print learning rate and global batch size information
    lr_initial = config.optimizer.learning_rate_scheduler.learning_rate
    lr_min = config.optimizer.learning_rate_scheduler.min_decay_lr
    lr_warmup_steps = config.optimizer.learning_rate_scheduler.lr_warmup_steps
    lr_decay_steps = config.optimizer.learning_rate_scheduler.lr_decay_steps
    lr_decay_start = config.optimizer.learning_rate_scheduler.lr_decay_starting_step
    lr_decay_style = config.optimizer.learning_rate_scheduler.lr_decay_style

    # Sample/Token per GPU (at once)
    bs_gpu_sample = config.tokens.micro_batch_size
    bs_gpu_token = bs_gpu_sample * config.tokens.sequence_length

    # Sample/Token in one step
    gbs_sample = bs_gpu_sample * config.parallelism.dp * config.tokens.batch_accumulation_per_replica
    gbs_token = gbs_sample * config.tokens.sequence_length

    total_tokens = config.tokens.train_steps * gbs_token
    total_tokens_billions = human_format(total_tokens).replace(".", ",")

    print(
            f"""
    🏋️  Model Parameters:
    ┌───────────────────────┬────────────────────────┐
    │ Total Parameters      │ {num_params:>22} │
    │ Layers                │ {config.model.model_config.num_hidden_layers:>22d} │
    │ Attention Heads       │ {config.model.model_config.num_attention_heads:>22d} │
    │ Hidden Size           │ {config.model.model_config.hidden_size:>22d} │
    │ Intermediate Size     │ {config.model.model_config.intermediate_size:>22d} │
    │ Context Length        │ {config.model.model_config.max_position_embeddings:>22d} │
    │ Tokenizer             │ {config.tokenizer.tokenizer_name_or_path[:22]:>22} │
    │ Vocab Size            │ {config.model.model_config.vocab_size:>22d} │
    └───────────────────────┴────────────────────────┘
    """
    )

    num_nodes = args.nodes if args.slurm else 1
    print(
            f"""
    🎛️ Parallelism Configuration:
    ┌───────────────────────┬────────────────────────┐
    │ Nodes                 │ {num_nodes:>22d} │
    │ Total GPUs            │ {config.parallelism.dp*config.parallelism.pp*config.parallelism.tp:>22d} │
    │ Data Parallel (DP)    │ {config.parallelism.dp:>22d} │
    │ Pipeline Parallel (PP)│ {config.parallelism.pp:>22d} │
    │ Tensor Parallel (TP)  │ {config.parallelism.tp:>22d} │
    └───────────────────────┴────────────────────────┘
    """
        )

    print(
            f"""
    📙 Training Configuration:
    ┌───────────────────────┬────────────────────────┐
    │ Total Tokens          │ {total_tokens_billions:>22} │
    │ Batch Size (per GPU)  │ {bs_gpu_token:>15,d} Tokens │
    │ Global Batch Size     │ {gbs_token:>15,d} Tokens │
    └───────────────────────┴────────────────────────┘
    """
        )

    print(
            f"""
    📊 Learning Rate Schedule:
    ┌───────────────────────┬────────────────────────┐
    │ Initial LR            │ {lr_initial:>22.2e} │
    │ Warmup Style          │ {config.optimizer.learning_rate_scheduler.lr_warmup_style[:22]:>22} │
    │ Warmup Steps          │ {lr_warmup_steps:>22d} │
    │ Decay Style           │ {lr_decay_style[:22]:>22} │
    │ Decay Start Step      │ {lr_decay_start:>22d} │
    │ Decay Steps           │ {lr_decay_steps:>22d} │
    │ Final LR              │ {lr_min:>22.2e} │
    └───────────────────────┴────────────────────────┘
    """
        )
       
    print(
            f"""
    🔧 Optimization Configuration:
    ┌───────────────────────┬────────────────────────┐
    │ Optimizer             │ {config.optimizer.optimizer_factory.__class__.__name__:>22} │
    │ Weight Decay          │ {config.optimizer.weight_decay:>22.2e} │
    │ Gradient Clipping     │ {config.optimizer.clip_grad:>22.2f} │
    │ Adam Epsilon          │ {config.optimizer.optimizer_factory.adam_eps:>22.2e} │
    │ Adam Beta1            │ {config.optimizer.optimizer_factory.adam_beta1:>22.2f} │
    │ Adam Beta2            │ {config.optimizer.optimizer_factory.adam_beta2:>22.2f} │
    │ ZeRO Stage            │ {config.optimizer.zero_stage:>22d} │
    │ FP32 Grad Accumulation│ {str(config.optimizer.accumulate_grad_in_fp32):>22} │
    └───────────────────────┴────────────────────────┘
    """
        )

    if config.checkpoints.checkpoints_path is None:
        config.checkpoints.checkpoints_path = str(
            Path(config.general.logs_path) / config.general.run / "checkpoints"
        )
        Path(config.checkpoints.checkpoints_path).mkdir(parents=True, exist_ok=True)

    config_path_yaml = str(Path(config.general.config_logs_path) / "launch_config.yaml")    
    config.save_as_yaml(config_path_yaml)
