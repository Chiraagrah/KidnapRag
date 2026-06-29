#!/bin/bash

## React용
CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_wo_sr

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_wo_sr


CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_qwen25.json --gpu_num 2 --dataset musique --type ours_wo_sr

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_only_query

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_no_trace

CUDA_VISIBLE_DEVICES=0,1 python /home/work/Redteaming/rag-exp/ReAct/generate_attack_doc_chanwoo_local_kill.py --input_path /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_seed1_clean_llama70b.json --gpu_num 2 --dataset musique --type ours_wo_sr
#####


########################################################################################################################


## 인덱싱: /Redteaming/rag-exp/

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_no_trace_납치4회_llama70b.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_no_trace_납치4회_llama70b \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_only_query_납치4회_llama70b.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_only_query_납치4회_llama70b \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_no_trace_납치5회_qwen25.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_no_trace_납치5회_qwen25 \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_only_query_납치5회_qwen25.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_only_query_납치5회_qwen25 \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25.jsonl \
    --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25 \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치4회_llama70b \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_ih_납치5회_qwen25.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_ih_납치5회_qwen25 \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

# CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
#     --corpus_path /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25.jsonl \
#     --output_dir /home/work/Redteaming/rag-exp/datasets/musique/REACT_CHANWOO_ours_wo_sr_납치5회_qwen25 \
#     --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6
# nohup bash /home/work/Redteaming/rag-exp/scripts/prepare_all_data.sh > PROCESS1.log 2>&1 &
# nohup bash /home/work/Redteaming/rag-exp/scripts/prepare_all_data2.sh > PROCESS2.log 2>&1 &