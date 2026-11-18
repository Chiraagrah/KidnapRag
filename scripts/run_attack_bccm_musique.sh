#!/bin/bash
# FULL-CHAIN BCCM attack on MuSiQue (Qwen2.5-32B) -- the ~10-doc/question baseline
# to compare against the single-link attack. Builds the poison e5 index from the
# existing REACT_BCCM_qwen25.jsonl (1,575 docs) then runs attack_react.py.
# Uses the DEDUPED clean corpus (same as the single-link runs) for apples-to-apples.
#
# Pre-req: REACT_BCCM_qwen25.jsonl must be on Wulver at datasets/musique/ (scp it up).
# Submit:  sbatch scripts/run_attack_bccm_musique.sh
# Output:  ReAct/results/adv_targeted_results/musique_seed1_BCCM_qwen25.json
#SBATCH --job-name=bccm_musique
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                 # vLLM Qwen on cuda:0, e5 retriever on cuda:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/logs/bccm_musique_%j.out
#SBATCH --error=/project/ss797/cs757/logs/bccm_musique_%j.err

set -euo pipefail
REPO=/project/ss797/cs757/KidnapRAG
ENV=KidnapRAG

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate "$ENV"
module load CUDA/12.8.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/project/ss797/cs757/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

cd "$REPO"
# 1. build the poison e5 index (1,575 docs; fast) -- the retriever asserts
#    corpus==index size, so this MUST exist and match the poison jsonl.
rm -rf ReAct/datasets/musique/REACT_BCCM_qwen25
python ReAct/scripts/build_index.py \
    --corpus_path ReAct/datasets/musique/REACT_BCCM_qwen25.jsonl \
    --output_dir  ReAct/datasets/musique/REACT_BCCM_qwen25 \
    --batch_size 256

# 2. run the full-chain BCCM attack (clean deduped corpus + poison corpus)
cd "$REPO/ReAct/ReAct"
python attack_react.py \
    --attack_method BCCM \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --seed 1 \
    --dataset musique \
    --max_steps 15
