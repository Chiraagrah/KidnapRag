#!/bin/bash
# Per-hop LEVERAGE probe on the MuSiQue clean subset (Qwen2.5-32B).
# Counterfactual chain-completion: corrupt each hop's sub-answer, see if the
# final answer flips. Needs a full A100 (32B via vLLM).
#
# Submit:  sbatch scripts/run_leverage_musique.sh
# Watch:   tail -f /project/ss797/cs757/logs/leverage_musique_<jobid>.out
#SBATCH --job-name=leverage_musique
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/logs/leverage_musique_%j.out
#SBATCH --error=/project/ss797/cs757/logs/leverage_musique_%j.err

set -euo pipefail

REPO=/project/ss797/cs757/KidnapRAG
ENV=KidnapRAG

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate "$ENV"
module load CUDA/12.8.0
# conda libstdc++ ahead of system /lib64 (vLLM import needs CXXABI_1.3.15)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/project/ss797/cs757/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1     # Qwen is cached; comment out if a load fails

cd "$REPO"
python scripts/leverage_probe.py \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --dev ReAct/datasets/musique/musique_ans_v1.0_dev.jsonl \
    --qid_to_idx ReAct/results/adv_targeted_results/musique_qid_to_idx.json \
    --seed 1 \
    --out ReAct/results/mis_diagnostic/musique_leverage.json
