#!/bin/bash
# Clean ReAct + ReliabilityRAG MIS diagnostic (attack_react_mis.py) against an
# already-running gpt-oss-120b server (ReliabilityRAG/scripts/serve_gpt_oss_local.slurm).
# Only the E5 retriever + DeBERTa NLI model run on this job's GPU; the agent
# LLM itself is called over HTTP, so one GPU is enough here.
#
# Normally launched indirectly via wulver_run_mhweb_with_server.sh, which
# waits for the server and fills in API_BASE before submitting this job.
# Can also be submitted directly once a server is confirmed up:
#   API_BASE="http://$(tr -d '[:space:]' < ~/vllm_local_server.txt)/v1" \
#     DATASET=mhweb_musique sbatch scripts/wulver_run_mhweb_mis.slurm
#
#SBATCH --job-name=mhweb_mis
#SBATCH --account=ss797
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mhweb_mis_%j.out
#SBATCH --error=logs/mhweb_mis_%j.err
set -euo pipefail

REPO="${REPO:-/project/ss797/dk694/Repos/Masters-Project-Reliability-RAG}"
DATASET="${DATASET:-mhweb_musique}"
PY="${PYTHON:-python3}"
: "${API_BASE:?set API_BASE to http://HOST:PORT/v1 (e.g. from ~/vllm_local_server.txt)}"
API_MODEL="${API_MODEL:-openai/gpt-oss-120b}"
MAX_STEPS="${MAX_STEPS:-15}"
SEED="${SEED:-1}"
NLI_DEVICE="${NLI_DEVICE:-cuda:0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "${CONDA_ENV:-}" ]; then
  source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

cd "$REPO/ReAct/ReAct"
echo "=== $(date -Is) running attack_react_mis.py: dataset=$DATASET api_base=$API_BASE model=$API_MODEL ==="
"$PY" attack_react_mis.py \
    --dataset "$DATASET" --seed "$SEED" --max_steps "$MAX_STEPS" \
    --api_base "$API_BASE" --api_model "$API_MODEL" --nli_device "$NLI_DEVICE" \
    $EXTRA_ARGS
