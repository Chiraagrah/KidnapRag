#!/bin/bash
#SBATCH --job-name=kidnaprag_smoke
#SBATCH --partition=debug_gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --account=ss797
#SBATCH --output=/project/ss797/cs757/KidnapRAG/logs/smoke_%j.out
#SBATCH --error=/project/ss797/cs757/KidnapRAG/logs/smoke_%j.err

source /apps/easybuild/el9_5.x86_64/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
conda activate KidnapRAG
export HF_HOME=/project/ss797/cs757/.cache/huggingface

cd /project/ss797/cs757/KidnapRAG/ReAct/ReAct
python -c "import torch, transformers, vllm; print('imports OK'); print('CUDA:', torch.cuda.is_available())"
python attack_react.py --help
ls -la ../datasets/hotpotqa/corpus.jsonl || echo "CORPUS MISSING"
ls -la ../datasets/hotpotqa/e5_index/ || echo "INDEX MISSING"
