#!/bin/bash
# Reuse (or start) the ReliabilityRAG gpt-oss-120b server, wait until it answers,
# then submit scripts/wulver_webthinker_exp.slurm with API_BASE set.
# Usage: [DATA=.. CORPUS=.. TAG=..] bash scripts/wulver_webthinker_with_server.sh [--launch-when-ready JOB]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
RRAG="${RRAG_DIR:-$(cd .. && pwd)/ReliabilityRAG}"
SERVER_FILE="${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
server_up() { [[ -s "$SERVER_FILE" ]] && curl -s --max-time 10 "http://$(tr -d '[:space:]' < "$SERVER_FILE")/v1/models" | grep -q '"id"'; }

if [[ "${1:-}" == "--launch-when-ready" ]]; then
  SERVER_JOB="${2:-}"; deadline=$(( $(date +%s) + WAIT_MINUTES*60 ))
  until server_up; do
    if [[ -n "$SERVER_JOB" ]]; then
      st="$(squeue -h -j "$SERVER_JOB" -o %T 2>/dev/null || true)"
      [[ "$st" =~ ^(RUNNING|CONFIGURING|PENDING|COMPLETING)$ ]] || { echo "server job gone ($st)" >&2; exit 1; }
    fi
    (( $(date +%s) > deadline )) && { echo "server not ready after ${WAIT_MINUTES} min" >&2; exit 1; }
    sleep 30
  done
  while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^(SLURM|SBATCH)_' || true)
  export API_BASE="http://$(tr -d '[:space:]' < "$SERVER_FILE")/v1"
  J=$(sbatch --parsable --export=ALL scripts/wulver_webthinker_exp.slurm)
  echo "Submitted experiment job ${J%%;*} (API_BASE=$API_BASE)"
  [[ "${STOP_SERVER:-0}" == "1" && -n "$SERVER_JOB" ]] && sbatch --dependency=afterany:"${J%%;*}" --account=ss797 --partition=general --qos=standard \
    --mem=1G --time=00:05:00 --wrap "scancel $SERVER_JOB" >/dev/null
  exit 0
fi

# Phase 1 (login node): dry checks, then queue server (if needed) + launcher
: "${GPT_OSS_DIR:?set GPT_OSS_DIR}"
DATA=${DATA:-datasets/browsecomp_plus/bcp_select100.json}; CORPUS=${CORPUS:-datasets/browsecomp_plus/bcp_dev_corpus.jsonl}
for f in "$DATA" "$CORPUS"; do [[ -s "$f" ]] || { echo "missing $f (run prepare_browsecomp_plus.py first)" >&2; exit 1; }; done
mkdir -p logs
QUEUED="$(squeue -h -u "$USER" -n gpt-oss-server -t PENDING,CONFIGURING,RUNNING -o '%i' 2>/dev/null | head -1 || true)"
DEP=(); SERVER_JOB=""
if server_up; then echo "Server up at $(cat "$SERVER_FILE") -- reusing."
elif [[ -n "$QUEUED" ]]; then SERVER_JOB="$QUEUED"; DEP=(--dependency=after:"$SERVER_JOB"); echo "Waiting for server job $SERVER_JOB"
else
  SERVER_JOB=$(cd "$RRAG" && MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" sbatch --parsable --export=ALL scripts/serve_gpt_oss_local.slurm); SERVER_JOB="${SERVER_JOB%%;*}"
  DEP=(--dependency=after:"$SERVER_JOB"); echo "Queued server job $SERVER_JOB"
fi
L=$(sbatch --parsable "${DEP[@]}" --job-name=KRAG-launcher --account=ss797 --partition=general --qos=standard \
  --cpus-per-task=1 --mem=1G --time=$(( WAIT_MINUTES + 15 )) --output=logs/launcher-%j.out --export=ALL \
  --wrap "bash scripts/wulver_webthinker_with_server.sh --launch-when-ready '$SERVER_JOB'")
echo "Queued launcher ${L%%;*}; follow: tail -f logs/launcher-${L%%;*}.out"
