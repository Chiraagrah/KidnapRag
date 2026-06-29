#!/bin/bash

## React용
CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_deepseek_32b.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_deepseek_32b.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_deepseek_32b.json --gpu_num 2 --dataset musique --type ours_wo_sr

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_qwq.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_qwq.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/musique/clean_qwq.json --gpu_num 2 --dataset musique --type ours_wo_sr


CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_deepseek_32b.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_only_query

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_deepseek_32b.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_no_trace

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_deepseek_32b.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_wo_sr

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_qwq.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_only_query

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_qwq.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_no_trace

CUDA_VISIBLE_DEVICES=2,3 python /home/work/Redteaming/Rag_uni/scripts/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/clean_qwq.json --gpu_num 2 --dataset 2wikimultihopqa --type ours_wo_sr




########################################################################################################################

## 인덱싱: /Redteaming/rag-exp/

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치4회_llama70b.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치4회_llama70b \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치5회_qwen25.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_ih_납치5회_qwen25 \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/2wikimultihopqa/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25 \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6
# nohup bash /home/work/Redteaming/rag-exp/scripts/prepare_all_data.sh > PROCESS1.log 2>&1 &