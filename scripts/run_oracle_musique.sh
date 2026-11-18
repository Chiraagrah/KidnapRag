#!/bin/bash
# ORACLE single-link attack on MuSiQue (Qwen2.5-32B). Builds the target list from
# the Wulver injectability + leverage results, then runs the agent with a poison
# forced into the target hop's observation. Needs a full A100.
#
# Submit:  sbatch scripts/run_oracle_musique.sh
# Watch:   tail -f /project/ss797/cs757/logs/oracle_musique_<jobid>.out
#SBATCH --job-name=oracle_musique
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/logs/oracle_musique_%j.out
#SBATCH --error=/project/ss797/cs757/logs/oracle_musique_%j.err

set -euo pipefail
REPO=/project/ss797/cs757/KidnapRAG
ENV=KidnapRAG
MIS=$REPO/ReAct/results/mis_diagnostic

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate "$ENV"
module load CUDA/12.8.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/project/ss797/cs757/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

# 1. (re)build the target list from the Wulver injectability + leverage results
python "$REPO/scripts/build_weaklink_targets.py" \
    --injectability "$MIS/musique_injectability.json" \
    --leverage      "$MIS/musique_leverage.json" \
    --out           "$MIS/musique_weaklink_targets.json"

# 2. oracle attack (clean retrieval + forced poison at the target hop)
cd "$REPO/ReAct/ReAct"
python oracle_singlelink.py \
    --model_path Qwen/Qwen2.5-32B-Instruct --dataset musique --max_steps 15 --seed 1 \
    --targets  "$MIS/musique_weaklink_targets.json" \
    --leverage "$MIS/musique_leverage.json" \
    --dev      "$REPO/ReAct/datasets/musique/musique_ans_v1.0_dev.jsonl" \
    --clean    "$MIS/musique_qwen25_mis_per_question.json" \
    --out      "$MIS/musique_oracle_singlelink.json"
