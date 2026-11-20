#!/bin/bash
# One command for a Wulver MH-Web ReAct + MIS run on your own gpt-oss-120b
# server:
#   1. reuses your server if it is already up -- or, if a server job is
#      already queued/loading, waits for that one -- and only otherwise
#      queues a new 72-hour server job in the sibling ReliabilityRAG repo
#      (scripts/serve_gpt_oss_local.slurm);
#   2. a small CPU "launcher" job starts as soon as the server job starts
#      running, waits until the model has loaded and answers, then submits
#      scripts/wulver_run_mhweb_mis.slurm (attack_react_mis.py) with API_BASE
#      filled in.
# The server is LEFT RUNNING afterwards for follow-up runs. Stop it yourself
# (scancel <server jobid>), or set STOP_SERVER=1 to cancel it automatically
# once this run's job ends.
# Everything waits inside Slurm, so you can log out right after running this.
#
# Usage:
#   bash scripts/wulver_run_mhweb_with_server.sh                       # dataset=mhweb_musique
#   DATASET=mhweb_musique_fq bash scripts/wulver_run_mhweb_with_server.sh
#   MAX_STEPS=20 SEED=2 bash scripts/wulver_run_mhweb_with_server.sh
#
# Follow it with:
#   squeue -u $USER                                  # gpt-oss-server, mhweb-launcher, mhweb_mis
#   tail -f logs/mhweb-with-server-<launcher jobid>.out
#
# Knobs (env): DATASET (default mhweb_musique), MAX_STEPS (15), SEED (1),
# API_MODEL (openai/gpt-oss-120b), NLI_DEVICE (cuda:0), EXTRA_ARGS (passed
# through to attack_react_mis.py, e.g. EXTRA_ARGS="--limit 10 --dry_run"),
# WAIT_MINUTES (how long the launcher waits for the model to load, default
# 90), STOP_SERVER=1, SERVER_TIME (server --time, default serve_gpt_oss_
# local.slurm's own 3-00:00:00 = 72h), RRAG_DIR (default ../ReliabilityRAG
# next to this repo).
# One-time setup in ReliabilityRAG first (if not already done):
#   bash scripts/download_gpt_oss.sh ; sbatch scripts/setup_vllm_env.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"                           # KidnapRag/
cd "$REPO"
RRAG_DIR="${RRAG_DIR:-$(cd "$REPO/.." && pwd)/Masters-Project-Reliability-RAG}"
[[ -d "$RRAG_DIR" ]] || { echo "ReliabilityRAG repo not found at $RRAG_DIR (set RRAG_DIR=)" >&2; exit 1; }

SERVER_FILE="${LOCAL_VLLM_SERVER_FILE:-$HOME/vllm_local_server.txt}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
DATASET="${DATASET:-mhweb_musique}"
MAX_STEPS="${MAX_STEPS:-15}"
SEED="${SEED:-1}"
API_MODEL="${API_MODEL:-openai/gpt-oss-120b}"
NLI_DEVICE="${NLI_DEVICE:-cuda:0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$REPO/ReAct/logs"

server_up() {   # true if the address file points at a server that answers
  [[ -s "$SERVER_FILE" ]] || return 1
  curl -s --max-time 10 "http://$(tr -d '[:space:]' < "$SERVER_FILE")/v1/models" | grep -q '"id"'
}

# ───────────── phase 2 (inside the launcher job): wait, then submit ─────────────
if [[ "${1:-}" == "--launch-when-ready" ]]; then
  SERVER_JOB="${2:-}"
  echo "=== $(date -Is) launcher on $(hostname): waiting up to ${WAIT_MINUTES} min for server job ${SERVER_JOB:-<existing>} ==="
  deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
  until server_up; do
    if [[ -n "$SERVER_JOB" ]]; then
      state="$(squeue -h -j "$SERVER_JOB" -o %T 2>/dev/null || true)"
      if [[ -z "$state" || ! "$state" =~ ^(RUNNING|CONFIGURING|PENDING|COMPLETING)$ ]]; then
        echo "server job $SERVER_JOB is no longer running (state: ${state:-gone}) -- see $RRAG_DIR/log/vllm-server-${SERVER_JOB}.err" >&2
        exit 1
      fi
    fi
    if (( $(date +%s) > deadline )); then
      echo "model not ready after ${WAIT_MINUTES} min -- see $RRAG_DIR/log/vllm-server-${SERVER_JOB}.out/.err; the server keeps running" >&2
      exit 1
    fi
    sleep 30
  done
  API_BASE="http://$(tr -d '[:space:]' < "$SERVER_FILE")/v1"
  echo "=== $(date -Is) server is up at $(cat "$SERVER_FILE"); submitting the experiment (dataset=$DATASET) ==="
  # submit as if from the login node: don't leak this launcher job's settings into the new job
  while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^(SLURM|SBATCH)_' || true)

  cd "$REPO"
  EXP_JOB=$(DATASET="$DATASET" MAX_STEPS="$MAX_STEPS" SEED="$SEED" API_MODEL="$API_MODEL" \
            NLI_DEVICE="$NLI_DEVICE" EXTRA_ARGS="$EXTRA_ARGS" API_BASE="$API_BASE" \
            sbatch --parsable --export=ALL scripts/wulver_run_mhweb_mis.slurm)
  EXP_JOB="${EXP_JOB%%;*}"
  echo "=== Submitted experiment job $EXP_JOB (API_BASE=$API_BASE); see ReAct/logs/mhweb_mis_${EXP_JOB}.out ==="

  if [[ -n "$SERVER_JOB" && "${STOP_SERVER:-0}" == "1" ]]; then
    stop_job=$(sbatch --parsable --dependency=afterany:"$EXP_JOB" --job-name=mhweb-stop-server \
      --account=ss797 --partition=general --qos=standard \
      --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
      --output="$REPO/ReAct/logs/stop-server-%j.out" \
      --wrap "echo stopping server job $SERVER_JOB after experiment job $EXP_JOB; scancel $SERVER_JOB")
    echo "=== server job $SERVER_JOB will be cancelled when job $EXP_JOB ends (stop job ${stop_job%%;*}) ==="
  else
    left="$( [[ -n "$SERVER_JOB" ]] && squeue -h -j "$SERVER_JOB" -o %L 2>/dev/null || true)"
    echo "=== server ${SERVER_JOB:+job $SERVER_JOB }keeps running${left:+ ($left left)} for follow-up runs; scancel it when done ==="
  fi
  exit 0
fi

# ───────────── phase 1 (you, on a login node): queue server + launcher ─────────────
mkdir -p "$REPO/ReAct/logs"

# an already queued/loading server job of yours (don't queue a second one)
QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n gpt-oss-server -t PENDING,CONFIGURING,RUNNING -o '%i %T' 2>/dev/null | head -1 || true)"
if server_up; then
  echo "Your server is already up at $(cat "$SERVER_FILE") -- reusing it."
  SERVER_JOB=""
  DEP=()
elif [[ -n "$QUEUED" ]]; then
  read -r SERVER_JOB state <<< "$QUEUED"
  echo "Server job $SERVER_JOB is already $state (not answering yet) -- waiting for it instead of queueing another."
  DEP=(--dependency=after:"$SERVER_JOB")
else
  TIME_ARGS=(); [[ -n "${SERVER_TIME:-}" ]] && TIME_ARGS=(--time="$SERVER_TIME")
  SERVER_JOB=$(cd "$RRAG_DIR" && sbatch --parsable "${TIME_ARGS[@]}" scripts/serve_gpt_oss_local.slurm)
  SERVER_JOB="${SERVER_JOB%%;*}"
  echo "Queued vLLM server job $SERVER_JOB for ${SERVER_TIME:-72 h} ($RRAG_DIR/log/vllm-server-$SERVER_JOB.out); it stays up for follow-up runs"
  DEP=(--dependency=after:"$SERVER_JOB")   # launcher starts once the server job has started
fi

LAUNCHER=$(sbatch --parsable "${DEP[@]}" --job-name=mhweb-launcher \
  --account=ss797 --partition=general --qos=standard \
  --ntasks=1 --cpus-per-task=1 --mem=1G --time=$(( WAIT_MINUTES + 15 )) \
  --output="$REPO/ReAct/logs/mhweb-with-server-%j.out" \
  --export=ALL \
  --wrap "WAIT_MINUTES='$WAIT_MINUTES' DATASET='$DATASET' MAX_STEPS='$MAX_STEPS' SEED='$SEED' API_MODEL='$API_MODEL' NLI_DEVICE='$NLI_DEVICE' EXTRA_ARGS='$EXTRA_ARGS' RRAG_DIR='$RRAG_DIR' STOP_SERVER='${STOP_SERVER:-0}' bash '$HERE/wulver_run_mhweb_with_server.sh' --launch-when-ready '${SERVER_JOB}'")
LAUNCHER="${LAUNCHER%%;*}"
echo "Queued launcher job $LAUNCHER: once the model answers it runs attack_react_mis.py --dataset $DATASET"
echo "Follow:  squeue -u \$USER   and   tail -f ReAct/logs/mhweb-with-server-$LAUNCHER.out"
