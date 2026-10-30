#!/bin/bash
#SBATCH --job-name=build_bccm_index
#SBATCH --partition=debug_gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/KidnapRAG/logs/build_bccm_%j.out
#SBATCH --error=/project/ss797/cs757/KidnapRAG/logs/build_bccm_%j.err

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate KidnapRAG
export HF_HOME=/project/ss797/cs757/.cache/huggingface

cd /project/ss797/cs757/KidnapRAG/ReAct
python scripts/build_index.py \
    --corpus_path /project/ss797/cs757/KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCM_qwen25.jsonl \
    --output_dir /project/ss797/cs757/KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCM_qwen25 \
    --model_name intfloat/e5-large-v2
