#!/bin/bash
#SBATCH --job-name=KRAG-misagree
#SBATCH --account=ss797
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/misagree_%j.out
#SBATCH --error=logs/misagree_%j.err
# Per-hop MIS / agreement diagnostic on a finished WebThinker results file.
# Env: API_BASE (required), RESULTS (default clean_gptoss.json), OUT, EXTRA_ARGS
# The NLI model (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) must be cached
# (download once on a login node), because HF_HUB_OFFLINE=1 here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME=${HF_HOME:-/project/ss797/dk694/.cache/huggingface} HF_HUB_OFFLINE=1
: "${API_BASE:?}"
RESULTS=${RESULTS:-Webthinker/results/browsecomp_plus/clean_gptoss.json}
OUT=${OUT:-Webthinker/results/browsecomp_plus/mis_agreement_clean.json}
mkdir -p logs
python3 Webthinker/scripts/mis_agreement.py --results "$RESULTS" --output "$OUT" \
  --api_base "$API_BASE" --model_name openai/gpt-oss-120b --nli_device cuda:0 --workers 8 ${EXTRA_ARGS:-}
