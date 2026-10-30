#!/bin/bash
# Quick smoke test: ReliabilityRAG MIS diagnostic on ONE question (--dry_run).
# Lightweight so it backfills into the queue fast.
#   sbatch run_mis_smoke.sh
#   tail -f /project/ss797/cs757/logs/mis_smoke_<jobid>.out
#SBATCH --job-name=mis_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                       # Qwen on cuda:1, NLI on cuda:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/logs/mis_smoke_%j.out
#SBATCH --error=/project/ss797/cs757/logs/mis_smoke_%j.err

set -euo pipefail

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate KidnapRAG
module load CUDA/12.8.0

# Put the conda env's newer libstdc++ ahead of the system /lib64 one,
# otherwise `import sqlite3` (via vllm->diskcache) fails with
# "CXXABI_1.3.15 not found" because module load prepends a system lib path.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_HOME=/project/ss797/cs757/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME

cd /project/ss797/cs757/KidnapRAG/ReAct/ReAct
python attack_react_mis.py \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --dataset hotpotqa --seed 1 \
    --nli_device cuda:0 --max_steps 15 --dry_run
