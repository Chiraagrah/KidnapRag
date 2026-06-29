#!/bin/bash
# Example script to run WebThinker agent

set -e

# Set Trap to kill background processes (vLLM launchers) on exit
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# Required arguments
DATA_PATH="${1:?Missing DATA_PATH}"
CORPUS_PATH="${2:?Missing CORPUS_PATH}"
OUTPUT_PATH="${3:?Missing OUTPUT_PATH}"

# Paths
RETRIEVAL_MODEL="intfloat/e5-large-v2" # Can be local path

# vLLM Configuration as provided by user
# Main Model: WebThinker (GPU 0)
MAIN_MODEL_PATH="Qwen/QwQ-32B"
MAIN_MODEL_NAME="QwQ-32B"
MAIN_PORT=8001
MAIN_GPU="0"

# Aux Model: Qwen (GPU 1)
AUX_MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
AUX_MODEL_NAME="Qwen2.5-32B-Instruct"
AUX_PORT=8002
AUX_GPU="1"

mkdir -p logs

echo "========================================================"
echo "Starting vLLM Orchestration for WebThinker"
echo "========================================================"
echo "[Script] DATA_PATH: $DATA_PATH"
echo "[Script] CORPUS_PATH: $CORPUS_PATH"
echo "[Script] POISON_CORPUS_PATH: $POISON_CORPUS_PATH"
echo "[Script] OUTPUT_PATH: $OUTPUT_PATH"

# Launch Main Model
python3 scripts/launch_vllm.py \
    --model "$MAIN_MODEL_PATH" \
    --served_model_name "$MAIN_MODEL_NAME" \
    --port "$MAIN_PORT" \
    --gpu_device "$MAIN_GPU" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 30000 \
    > logs/vllm_main.log 2>&1 &
PID_MAIN=$!
echo "[Script] Launched Main Model (PID $PID_MAIN). Logs: logs/vllm_main.log"

# Launch Aux Model
python3 scripts/launch_vllm.py \
    --model "$AUX_MODEL_PATH" \
    --served_model_name "$AUX_MODEL_NAME" \
    --port "$AUX_PORT" \
    --gpu_device "$AUX_GPU" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 30000 \
    > logs/vllm_aux.log 2>&1 &
PID_AUX=$!
echo "[Script] Launched Aux Model (PID $PID_AUX). Logs: logs/vllm_aux.log"

# Wait for Servers
echo "[Script] Waiting for servers to be ready..."
python3 scripts/wait_for_servers.py \
    --urls "http://localhost:$MAIN_PORT/v1" "http://localhost:$AUX_PORT/v1" \
    --timeout 600

if [ $? -ne 0 ]; then
    echo "[Script] Failed to start servers. Exiting."
    exit 1
fi

echo "[Script] Servers Ready! Starting Attack..."

CUDA_VISIBLE_DEVICES=0,1,2 python3 scripts/run_attack.py \
    --model webthinker \
    --api_base_url "http://localhost:$MAIN_PORT/v1" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --aux_api_base_url "http://localhost:$AUX_PORT/v1" \
    --aux_model_path "$AUX_MODEL_PATH" \
    --aux_model_name "$AUX_MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_search_limit 10 \
    --concurrent_limit 32

echo "[Script] Finished. Cleaning up..."