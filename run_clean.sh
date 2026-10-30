#!/bin/bash
#SBATCH --job-name=kidnaprag_clean
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ss797
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/ss797/cs757/KidnapRAG/logs/clean_%j.out
#SBATCH --error=/project/ss797/cs757/KidnapRAG/logs/clean_%j.err

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate KidnapRAG
module load CUDA/12.8.0
export HF_HOME=/project/ss797/cs757/.cache/huggingface

cd /project/ss797/cs757/KidnapRAG/ReAct/ReAct
CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react.py \
    --attack_method clean \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --seed 1 \
    --dataset hotpotqa
