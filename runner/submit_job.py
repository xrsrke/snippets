# This script should create the experiment folder with the name of wandb project
import argparse
import os
from datetime import datetime
import subprocess

from dataclasses import dataclass

def set_system_path():
    import sys
    from pathlib import Path
    import importlib
    
    package = importlib.import_module("nanotron")
    # NOTE:  Path(package.__file__).parent = .../nanotron/src/nanotron
    # we want .../nanotron
    package_path = Path(package.__file__).parent.parent.parent
    sys.path.append(str(package_path))


# from brrr.config.brrr_config import BrrrConfig

# set_system_path()

# from examples.doremi.doremi.config import DoReMiArgs


# @dataclass(kw_only=True)  # pylint: disable=unexpected-keyword-arg
# class DoReMiConfig(BrrrConfig):
#     """Main configuration class"""

#     doremi: DoReMiArgs
    
def generate_training_slurm_script(
    nodes,
    nproc_per_node,
    output_path,
    config_file,
    brrr_repo_path,
    conda_path,
    is_debug=False,
    script_path="use_trainer.py"
):
    launcher_content_list = []
    
    if is_debug:
        for i in range(nodes):
            launcher_content_list.append(
                f'export LAUNCHER{i}="debugpy-run -p 1234 -m torch.distributed.run \\\n'
                '   -- \\\n'
                f'  --node_rank {i} \\\n'
                '   --max_restarts 0 \\\n'
                f'  --nproc_per_node {nproc_per_node} \\\n'
                f'  --nnodes {nodes} \\\n'
                '   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\\n'
                '   --rdzv_backend c10d"'
                '   --tee 3 \\\n'
            )
    else:
        launcher_content_list.append(
            'export LAUNCHER="python -u -m torch.distributed.run \\\n'
            f'    --nproc_per_node {nproc_per_node} \\\n'
            f'    --nnodes {nodes} \\\n'
            '    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\\n'
            '    --rdzv_backend c10d \\\n'
            '    --tee 3"\n\n'
        )
        
    # NOTE: generate a random port from the allowed range
    import random
    random_port = random.randint(6000, 7000)
        
    #TODO: Do Slurm Job arrays
    slurm_template = (
        f'#!/bin/bash\n'
        'set -x -e\n\n'
        
        'source ~/.bashrc\n'
        'export AWS_DEFAULT_REGION=us-east-1\n'
        'export USE_FAST=1\n'
        'export CUDA_DEVICE_MAX_CONNECTIONS=1\n\n'
        
        
        f'BRRR_REPO={brrr_repo_path}\n'
        'MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)\n'
        f'MASTER_PORT={random_port}\n'
        'USE_SYSTEM_NCCL=1\n\n'
        
        # f'source {conda_path}/etc/profile.d/conda.sh\n'
        # f'conda activate {conda_path}/envs/env-brrr\n'

        # f'conda init bash\n'
        f'source /admin/home/phuc_nguyen/miniconda3/etc/profile.d/conda.sh\n'
        f'conda activate {conda_path}\n'


        f"source /etc/profile.d/modules.sh\n\n"
        'module load cuda/12.1\n\n'
        
        'echo "START TIME: $(date)"\n\n'
        'echo "Git commit: $(git rev-parse HEAD)"\n\n'
        'echo "printenv:"\n'
        'printenv\n\n'
        'echo "nvidia-smi:\"\n'
        'nvidia-smi\n\n'
        'echo "torch version:"\n'
        
        'python -m torch.utils.collect_env\n\n'
        
        f'CMD="{brrr_repo_path}/{script_path} \\\n'
        f'   --config-file {config_file}"\n\n'
    )
    
    if is_debug:
        for i in range(nodes):
            slurm_template += (
                f'{launcher_content_list[i]}\n'
            )
    else:
        slurm_template += f'{launcher_content_list[0]}\n'

    slurm_template += (        
        'echo $CMD\n'
        'export NCCL_ASYNC_ERROR_HANDLING=1\n'
        'export NCCL_DEBUG=WARN\n'
        'export NCCL_DEBUG_SUBSYS=COLL\n\n'
        'echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"\n'
        'echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"\n\n'
        
        # Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
        'random_milliseconds=$(( RANDOM % 1001 ))\n'
        'sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")\n'
        'echo "Sleeping for $sleep_time seconds..."\n'
        'sleep $sleep_time\n\n'
    )
    
    if is_debug:
        for i in range(nodes):
            slurm_template += (
                f'srun -u bash -c "$LAUNCHER{i} --node_rank $SLURM_PROCID --role $SLURMD_NODENAME $CMD"\n'
            )
    else:
        slurm_template += (
            'srun -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME $CMD"\n'
        )
    
    slurm_template += ('echo "END TIME: $(date)"\n')


    print(f"Generating training SLURM script to {output_path}\n")
    # Write the SLURM script to a file
    with open(output_path, 'w') as file:
        file.write(slurm_template)

def generate_lighteval_jinja_script(
    nproc_per_node,
    experiment_path,
    brrr_repo_path,
    conda_path,
    hf_cache_path,
    is_debug=False
):

    if is_debug:
        raise NotImplementedError("Debug mode for LightEval is not implemented yet")
        launcher_content = (
            'export LAUNCHER="debugpy-run -p 1234 -m torch.distributed.run \\\n'
            '   -- \\\n'
            f'  --nproc_per_node {nproc_per_node} \\\n'
            '   --nnodes $COUNT_NODE \\\n'
            '   --max_restarts 0 \\\n'
            '   --tee 3 \\\n'
            '    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\\n'
            '   --rdzv_backend c10d"'
        )        
    else:
        launcher_content = (
            'export LAUNCHER="python -u -m torch.distributed.run \\\n'
            f'   --nproc_per_node {nproc_per_node} \\\n'
            '   --nnodes $COUNT_NODE \\\n'
            '   --max_restarts 0 \\\n'
            '   --tee 3"'
        )
            
    jinja_template = (
        f'#!/bin/bash\n'
        '#SBATCH --job-name={{ slurm_name }}\n'
        '#SBATCH --nodes=1\n'
        '#SBATCH --ntasks-per-node=1\n'
        '#SBATCH --partition=hopper-prod\n'
        '#SBATCH --gres=gpu:8\n'
        '#SBATCH --cpus-per-task=32\n'
        f'#SBATCH --output={experiment_path}/logs/%x-%n-%j.out\n'
        f'#SBATCH --error={experiment_path}/logs/%x-%n-%j.out\n'
        'set -x -e\n\n'
        
        f'BRRR_REPO={brrr_repo_path}\n'
        f'S3_FOLDER={experiment_path}/lighteval/s3_tmp\n\n'

        f'echo "START TIME: $(date)"\n'

        f'source /admin/home/phuc_nguyen/miniconda3/etc/profile.d/conda.sh\n'
        f'conda activate {conda_path}\n'
       
        f'echo python3 version = $(python3 --version)\n\n'
        
        f'# SLURM stuff\n'
        f'export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`\n'
        f'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)\n'
        f"export MASTER_PORT=6000\n"
        f'export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`\n\n'

        f"export TMPDIR=/scratch\n"
        f'export HF_DATASETS_CACHE="{hf_cache_path}"\n'
        f'export CUBLAS_WORKSPACE_CONFIG=":4096:8"\n'
        f'export CUDA_DEVICE_MAX_CONNECTIONS="1"\n\n'
        'echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"\n'
        'echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"\n\n'
        
       
        f"source /etc/profile.d/modules.sh\n\n"
        "module load cuda/12.1\n\n"
        
        f'# Hugging Face token\n'
        f'if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then\n'
        f'  # Attempt to read the token from the cache\n'
        f'  if TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null); then\n'
        f'    export HUGGING_FACE_HUB_TOKEN=$TOKEN\n'
        f'  else\n'
        f'    echo "Error: The environment variable HUGGING_FACE_HUB_TOKEN is not set and the token cache could not be read."\n'
        f'    exit 1\n'
        f'  fi\n'
        f'fi\n\n'
        
        f'echo go $COUNT_NODE\n'
        f'echo $HOSTNAMES\n'
        f'# Copying checkpoint from s3 to the node on node\n'
        's5cmd cp {{ model_checkpoint_path }}* $S3_FOLDER\n\n'
        
        f'CMD="$BRRR_REPO/run_evals_nanotron.py \\\n'
        '    --checkpoint-config-path ${S3_FOLDER}/config.yaml"\n\n'
        
        f'{launcher_content}'
        '\n'
        
        'srun -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME $CMD"'
    )

    with open(f"{experiment_path}/run_eval.slurm.jinja", 'w') as f:
        f.write(jinja_template)

    print(f"Lighteval script generated at '{experiment_path}/run_eval.slurm.jinja'\n")
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True, help="Path to the config file")
    args.add_argument("--no_training", action="store_false", help="Run training")
    args.add_argument("--use_lighteval", action="store_true", help="Use lighteval")
    args.add_argument("--debug_train", action="store_true", help="Debug mode")
    args.add_argument("--no_wandb", action="store_true", help="Do not use wandb")
    # Default
    args.add_argument("--brrr_repo_path", type=str, default="/fsx/phuc/projects/reference/brrr", help="Path to the brrr repo")
    args.add_argument("--script_path", type=str, default="use_trainer.py", help="Path to the brrr repo")
    args.add_argument("--is_brrr_config", type=str, default="true", help="")
    args.add_argument("--conda_path", type=str, default="/fsx/phuc/projects/reference/env/", help="Path to the conda environment")
    args.add_argument("--hf_cache_path", type=str, default="/fsx/phuc/.cache/huggingface_cache", help="Path to the huggingface cache")
    # Slurm
    args.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    args.add_argument("--nproc_per_node", type=int, default=1, help="Number of gpus per node")
    args.add_argument("--nodelist", type=str, default=None, help="List of nodes")
    args.add_argument("--wait_job_id", type=str, default=None, help="Wait for the job id to finish")
    args.add_argument("--ignore-nodes", type=str, default=None, help="Nodes to ignore, e.g. 'ip-26-0-167-[217,245]'")
    args.add_argument("--exclusive", action="store_true", help="Request exclusive access to nodes")


    args = args.parse_args()

    if args.is_brrr_config == "true":
        from brrr.config import (
            BrrrConfig,
            get_config_from_file
        )
        config = get_config_from_file(args.config, config_class=BrrrConfig)
    elif args.is_brrr_config == "false":
        from nanotron.config import Config, get_config_from_file
        config = get_config_from_file(args.config, config_class=Config)
    else:
        raise ValueError("is_brrr_config must be either 'true' or 'false'")
    
    if config.lighteval is None and args.use_lighteval:
        raise ValueError("You cannot use lighteval without lighteval config in the config file")
    
    if args.is_brrr_config == "true":
        # out_dir_path = f"/fsx/phuc/new_workspace/experiments/{config.experiment_logger.wandb_logger.wandb_project}"
        out_dir_path = f"/fsx/phuc/new_workspace/experiments/fp8_for_nanotron"
    elif args.is_brrr_config == "false":
        out_dir_path = f"/fsx/phuc/new_workspace/experiments/{config.general.project}"
    else:
        raise ValueError("is_brrr_config must be either 'true' or 'false'")

    config_name = os.path.basename(args.config)
    config_name = os.path.splitext(config_name)[0]
    output_log_dir = f"{out_dir_path}/{config_name}" 
    
    directories = [
        out_dir_path,
        output_log_dir,
        f"{output_log_dir}/checkpoints",
        f"{output_log_dir}/configs",
        f"{output_log_dir}/logs",
        f"{output_log_dir}/logs/tb_logs",
        f"{output_log_dir}/lighteval",
        f"{output_log_dir}/lighteval/s3_tmp",
        f"{output_log_dir}/lighteval/slurm_scripts"
    ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
                
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yaml_config_output_path = os.path.join(output_log_dir, "configs", f"config_{datetime_now}.yaml")
    slurm_script_output_path = os.path.join(output_log_dir, f"train_{datetime_now}.slurm")
    
    print(f"Copying config file to {yaml_config_output_path}\n")
    subprocess.run(['cp', args.config, yaml_config_output_path], check=True)
    
    if args.no_training:
        # Generate SLURM script
        generate_training_slurm_script(
            nodes=args.nodes,
            nproc_per_node=args.nproc_per_node,
            output_path=slurm_script_output_path,
            config_file=yaml_config_output_path,
            brrr_repo_path=args.brrr_repo_path,
            conda_path=args.conda_path,
            is_debug=args.debug_train,
            script_path=args.script_path
        )
    
    if args.use_lighteval:
        generate_lighteval_jinja_script(
            nproc_per_node=args.nproc_per_node,
            experiment_path=output_log_dir,
            brrr_repo_path=args.brrr_repo_path,
            conda_path=args.conda_path,
            hf_cache_path=args.hf_cache_path,
        )

    env_vars = os.environ.copy()
    env_vars["USE_WANDB"] = "0" if args.no_wandb else "1"

    if args.nodelist is not None:
        env_vars["SLURM_NODELIST"] = args.nodelist

    sbatch_command = [
        'sbatch',
        f'--job-name={config_name}',
        f'--nodes={args.nodes}',
        '--ntasks-per-node=1',
        # '--cpus-per-task=96',
        f'--gres=gpu:h100:{args.nproc_per_node}',
        '--mem-per-cpu=11G',
        '--partition=hopper-prod',
        '--array=1-100%1', # create a job array with 100 tasks and run them one by one
        f'--output={out_dir_path}/{config_name}/logs/train-%n-%j.out',
        f'--error={out_dir_path}/{config_name}/logs/train-%n-%j.out',
        # '--dependency=afterany:5535476_1',  # Run after job 5535476_1 finishes or is cancelled
        '--qos=normal',
        slurm_script_output_path
    ]

    if args.exclusive:
        sbatch_command.insert(-1, '--exclusive')

    if args.ignore_nodes:
        sbatch_command.insert(-1, f'--exclude={args.ignore_nodes}')

    if args.wait_job_id is not None:
        # NOTE: afterany: The new job should be launched after the specified job(s) have terminated, regardless of their exit status.
        # afternotok: The new job should be launched after the specified job(s) have terminated with a non-zero exit code.
        sbatch_command.insert(-1, f'--dependency=afterany:{args.wait_job_id}')
    
    print(f"Sbatch command: {' '.join(sbatch_command)}")
    
    print(
        f"Running {slurm_script_output_path} on {args.nodes} "
        f"node(s) of {args.nproc_per_node} "
        f"gpu(s) each with config file "
        f"{yaml_config_output_path}\n"
    )

    # # Create the bash command
    # bash_command = f"source ~/.bashrc; {' '.join(sbatch_command)}"

    # # Create the full command with env -i
    # full_command = ["env", "-i", "bash", "-c", bash_command]

    # First, find the full path to sbatch
    sbatch_path = subprocess.run(['which', 'sbatch'], capture_output=True, text=True).stdout.strip()

    # Create the bash command
    bash_command = f"source ~/.bashrc; export PATH=$PATH:{os.path.dirname(sbatch_path)}; {' '.join(sbatch_command)}"

    # Create the full command with env -i
    full_command = ["env", "-i", "bash", "-c", bash_command]

    # Run the command
    subprocess.run(full_command, check=True, env=env_vars)