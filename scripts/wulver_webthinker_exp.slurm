#!/bin/bash
#SBATCH --job-name=KRAG-bcp
#SBATCH --account=ss797
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bcp_%j.out
#SBATCH --error=logs/bcp_%j.err
# Experiment job: 1 GPU for E5 retrieval; the LLM runs on the gpt-oss server (API_BASE).
# Env: API_BASE (required), GPT_OSS_DIR, DATA, CORPUS, POISONED_CORPUS, TAG, EXTRA_ARGS
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME=${HF_HOME:-/project/ss797/cs757/.cache/huggingface} HF_HUB_OFFLINE=1
: "${API_BASE:?}" "${GPT_OSS_DIR:?}"
DATA=${DATA:-datasets/browsecomp_plus/bcp_select100.json}
CORPUS=${CORPUS:-datasets/browsecomp_plus/bcp_dev_corpus.jsonl}
TAG=${TAG:-clean_gptoss}
POISON=(); [[ -n "${POISONED_CORPUS:-}" ]] && POISON=(--poisoned_corpus_path "$POISONED_CORPUS")
mkdir -p logs Webthinker/results/browsecomp_plus
python3 Webthinker/scripts/run_attack.py --model webthinker \
  --api_base_url "$API_BASE" --model_name openai/gpt-oss-120b --model_path "$GPT_OSS_DIR" \
  --aux_api_base_url "$API_BASE" --aux_model_name openai/gpt-oss-120b --aux_model_path "$GPT_OSS_DIR" \
  --data_path "$DATA" --corpus_path "$CORPUS" "${POISON[@]}" \
  --retriever_device cuda:0 --max_chunks_per_doc 1 \
  --max_search_limit 10 --top_k 5 --concurrent_limit 16 \
  --output_path "Webthinker/results/browsecomp_plus/${TAG}.json" ${EXTRA_ARGS:-}
