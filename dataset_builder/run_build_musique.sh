#!/usr/bin/env bash
# Full MH-Web build for MuSiQue (plan doc claude/multihop-web-dataset-plan.md,
# stages B-E). Run from anywhere; paths are relative to this script.
#
#   bash run_build_musique.sh            # all steps
#   bash run_build_musique.sh collect    # only one step: hops|collect|label|build|test
#
# Step "collect" needs outbound HTTPS to api.keenable.ai (your own machine or
# a Wulver LOGIN node -- compute nodes are offline) and reads the key from
# dataset_builder/.env. It is resumable: rerun the same command after an
# interruption and it continues where it stopped, without paying twice.
# Every other step is offline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"                       # KidnapRag/
PY="${PYTHON:-python3}"
BUILD="$HERE/build/musique"
RAW="$REPO/ReAct/dataset/musique/musique_ans_v1.0_dev.jsonl"
[ -f "$RAW" ] || RAW="$REPO/ReAct/datasets/musique/musique_ans_v1.0_dev.jsonl"
QMAP="$REPO/ReAct/results/adv_targeted_results/musique_qid_to_idx.json"
ADV="$REPO/ReAct/results/adv_targeted_results/musique.json"
NAME="${NAME:-mhweb_musique}"
RESERVE="${RESERVE:-20}"
MAX_CALLS="${MAX_CALLS:-2500}"       # budget guard on NEW calls per run (cached searches are free)
N="${N:-100}"                        # questions to select into the corpus
STEP="${1:-all}"
mkdir -p "$BUILD"

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'
[ -f "$RAW" ] || { echo "raw MuSiQue dev file not found: $RAW" >&2; exit 1; }

run() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

if run test; then
  echo "== offline tests"
  "$PY" "$HERE/test_builder.py" > "$BUILD/test.log" 2>&1 && tail -1 "$BUILD/test.log" \
    || { cat "$BUILD/test.log"; exit 1; }
fi

if run hops; then
  echo "== 01 build hops (100 paper targets + $RESERVE reserves)"
  if [ -f "$BUILD/queries_musique.json" ]; then
    # Grow the existing file instead of resampling, so already-collected
    # searches stay valid (01_build_hops.py --extend-from).
    cp "$BUILD/queries_musique.json" "$BUILD/queries_musique.prev.json"
    "$PY" "$HERE/01_build_hops.py" --raw "$RAW" --qid-map "$QMAP" \
        --out "$BUILD/queries_musique.json" --reserve "$RESERVE" \
        --extend-from "$BUILD/queries_musique.prev.json"
  else
    "$PY" "$HERE/01_build_hops.py" --raw "$RAW" --qid-map "$QMAP" \
        --out "$BUILD/queries_musique.json" --reserve "$RESERVE"
  fi
fi

if run collect; then
  echo "== 02 Keenable collection (resumable; cap $MAX_CALLS new calls)"
  "$PY" "$HERE/02_keenable_collect.py" --input "$BUILD/queries_musique.json" \
      --out "$BUILD/searches_musique.json" --cache-dir "$HERE/keenable_raw" \
      --max-results 50 --snippet-max-length 2000 --mode pro --max-calls "$MAX_CALLS"
  "$PY" - "$BUILD/searches_musique.json" "$BUILD/queries_musique.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); q = json.load(open(sys.argv[2]))
need = sum(1 + 2 * len(e["hops"]) for e in q)
ok = sum(1 for x in s if not x.get("error"))
print(f"collection status: {ok}/{need} searches OK, {sum(1 for x in s if x.get('error'))} errored")
if ok < need:
    print("not complete -- rerun this step before labelling", file=sys.stderr); sys.exit(2)
EOF
fi

if run label; then
  echo "== 04 clean / dedup / label (G1 gate)"
  "$PY" "$HERE/04_label_relevance.py" --queries "$BUILD/queries_musique.json" \
      --searches "$BUILD/searches_musique.json" --out-dir "$BUILD/labeled"
fi

if run build; then
  echo "== 05 corpus + ReAct files (hop searches only = default corpus)"
  "$PY" "$HERE/05_build_corpus.py" --queries "$BUILD/queries_musique.json" \
      --labeled-dir "$BUILD/labeled" --raw "$RAW" --adv-targets "$ADV" \
      --repo-root "$REPO" --name "$NAME" --n "$N"
  echo "== 05 ablation corpus (with full-question results)"
  "$PY" "$HERE/05_build_corpus.py" --queries "$BUILD/queries_musique.json" \
      --labeled-dir "$BUILD/labeled" --raw "$RAW" --adv-targets "$ADV" \
      --repo-root "$REPO" --name "${NAME}_fq" --n "$N" --include-full-question
  echo
  echo "Next: copy ReAct/datasets/$NAME* to Wulver and run"
  echo "  NAME=$NAME sbatch scripts/wulver_build_mhweb_index.slurm"
fi
