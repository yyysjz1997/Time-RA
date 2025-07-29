#!/bin/bash

model_paths_wo_finetuning=(
    "/path/to/model/deepseek-llm-7b-chat"
    "/path/to/model/gemma-2-2b-it"
    "/path/to/model/gemma-7b-it"
    "/path/to/model/llama-3-8b-instruct"
    "/path/to/model/llama-3.2-3B-Instruct"
    "/path/to/model/phi-4-mini-instruct"
    "/path/to/model/qwen2.5-3b-instruct"
    "/path/to/model/qwen2.5-7b-instruct"
)

model_names=(
    "DeepSeek-7b"
    "gemma-2-2b"
    "gemma-7b"
    "llama-3-8b"
    "llama-3.2-3B"
    "phi-4-mini"
    "qwen2.5-3b"
    "qwen2.5-7b"
)

for i in "${!model_names[@]}"; do
    model_path_wo_finetuning="${model_paths_wo_finetuning[$i]}"
    model_name="${model_names[$i]}"
    
    echo "==============================================="
    echo "Running model_path: $model_path_wo_finetuning | model_name: $model_name"
    echo "==============================================="

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_anomaly_multiclf_reason.py --model_path "$model_path_wo_finetuning"  --model_name "$model_name"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_anomaly_multiclf_reason.py --model_path "$model_path_wo_finetuning"  --model_name "$model_name"
    
    echo "Finished running $model_path_wo_finetuning with $model_name"
    echo
done
