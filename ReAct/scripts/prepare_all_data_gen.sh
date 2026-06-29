#!/bin/bash

## 공격성능평가
## Redteaming/rag-exp/ReAct
# reminder, instructional, sandwich, spotlight
## Llama

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight


CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ours납치4회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ours납치4회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique --defense_method spotlight


CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight


CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ours납치3회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho_defense.py --attack_method ours납치3회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_no_trace_납치4회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_only_query_납치4회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_wo_sr_납치4회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset musique


CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_no_trace_납치3회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_only_query_납치3회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa

CUDA_VISIBLE_DEVICES=0,1,2,3 python attack_react_so_kyuho.py --attack_method ours_wo_sr_납치3회  --model_path /home/work/chanwoo/model_hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --seed 1 --dataset 2wikimultihopqa
##Qwen25
## 먼저 비교할 거: completion_real

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight


CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ours납치5회 --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ours납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method naive  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ignore  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_real  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method completion_realcmb  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method poisonedRAG  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method topicattack  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method paradox  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ours납치5회 --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method sandwich

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho_defense.py --attack_method ours납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa --defense_method spotlight

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_no_trace_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_only_query_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_wo_sr_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset musique

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_no_trace_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_only_query_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa

CUDA_VISIBLE_DEVICES=0,1 python attack_react_so_kyuho.py --attack_method ours_wo_sr_납치5회  --model_path /home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --seed 1 --dataset 2wikimultihopqa

# nohup bash /home/work/Redteaming/rag-exp/scripts/prepare_all_data_gen.sh > PROCESS_GEN.log 2>&1 &