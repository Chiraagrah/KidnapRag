#!/bin/bash
# E5-large-v2 index for an MH-Web corpus (plan doc stage E).
#   NAME=mhweb_musique sbatch scripts/wulver_build_mhweb_index.slurm
#   NAME=mhweb_musique_fq sbatch scripts/wulver_build_mhweb_index.slurm
# ~30k snippets: a few minutes on one GPU. Compute nodes are offline, so cache
# intfloat/e5-large-v2 on a login node first (HF_HUB_DISABLE_XET=1 huggingface-cli download ...).
#SBATCH --job-name=mhweb_index
#SBATCH --account=ss797
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mhweb_index_%j.out
#SBATCH --error=logs/mhweb_index_%j.err
set -euo pipefail

REPO="${REPO:-/project/ss797/dk694/Repos/Masters-Project-Reliability-RAG}"
NAME="${NAME:-mhweb_musique}"
PY="${PYTHON:-python3}"
export HF_HUB_OFFLINE=1
[ -n "${HF_HOME:-}" ] || export HF_HOME="/project/ss797/dk694/.cache/huggingface"
if [ -n "${CONDA_ENV:-}" ]; then
  source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

CORPUS="$REPO/ReAct/datasets/$NAME/corpus.jsonl"
OUT="$REPO/ReAct/datasets/$NAME/e5_index"
[ -f "$CORPUS" ] || { echo "missing $CORPUS" >&2; exit 1; }
echo "corpus: $(wc -l < "$CORPUS") rows -> $OUT"

cd "$REPO/ReAct"
"$PY" scripts/build_index.py --corpus_path "$CORPUS" --output_dir "$OUT" \
    --model_name "${E5_MODEL:-intfloat/e5-large-v2}" --batch_size 256
ls -la "$OUT"
