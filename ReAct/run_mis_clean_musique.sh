#!/bin/bash
# ReliabilityRAG MIS diagnostic on the CLEAN (no-attack) ReAct run -- MuSiQue.
# MuSiQue (2-4 hop) + Qwen2.5-32B-Instruct, NJIT Wulver (account ss797 / user cs757).
#
# NOTE: run this on a PROPER corpus rebuilt from the official MuSiQue source (deduped),
# placed at datasets/musique/{corpus.jsonl,e5_index}. The KidnapRAG-shipped corpus is
# ~56% exact-duplicate passages (21,100 unique of 48,315 rows), which distorts the MIS
# (identical docs never contradict -> inflated MIS -> understated weakness).
#
# Submit:  sbatch run_mis_clean_musique.sh
# Watch:   squeue -u cs757     tail -f /project/ss797/cs757/logs/mis_clean_musique_<jobid>.out
#SBATCH --job-name=mis_clean_musique
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                       # Qwen on cuda:1, NLI on cuda:0 (smoke-verified)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00                    # QOS 'standard' cap is 72h
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/logs/mis_clean_musique_%j.out
#SBATCH --error=/project/ss797/cs757/logs/mis_clean_musique_%j.err

set -euo pipefail

# ---- where you cloned KidnapRAG on Wulver (EDIT if different) --------------
REPO=/project/ss797/cs757/KidnapRAG
# ---- the conda env you used for your KidnapRAG clean/BCCM runs (EDIT) ------
ENV=KidnapRAG

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate "$ENV"
module load CUDA/12.8.0

# Put the conda env's newer libstdc++ ahead of the system /lib64 one,
# otherwise `import sqlite3` (via vllm->diskcache) fails with
# "CXXABI_1.3.15 not found" because module load prepends a system lib path.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# keep all model caches on /project (HOME is 50GB-quota'd)
export HF_HOME=/project/ss797/cs757/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1     # Qwen + DeBERTa NLI are cached; comment out if a load fails on a missing file

cd "$REPO/ReAct/ReAct"
python attack_react_mis.py \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --dataset musique \
    --seed 1 \
    --nli_device cuda:0 \
    --max_steps 15
