#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_names=(
    "/path/to/models/deepseek-llm-7b-chat"
    "/path/to/models/gemma-2-2b-it"
    "/path/to/models/gemma-7b-it"
    "/path/to/models/llama-3-8b-instruct"
    "/path/to/models/llama-3.2-3B-Instruct"
    "/path/to/models/phi-4-mini-instruct"
    "/path/to/models/qwen2.5-3b-instruct"
    "/path/to/models/qwen2.5-7b-instruct"
)

save_paths=(
    "outputs/DeepSeek-7b-tsad-lora"
    "outputs/gemma-2-2b-tsad-lora"
    "outputs/gemma-7b-tsad-lora"
    "outputs/llama-3-8b-tsad-lora"
    "outputs/llama-3.2-3B-tsad-lora"
    "outputs/phi-4-mini-tsad-lora"
    "outputs/qwen2.5-3b-tsad-lora"
    "outputs/qwen2.5-7b-tsad-lora"
)

for ((i=0; i<${#model_names[@]}; i++))
do
    model_name=${model_names[$i]}
    save_path=${save_paths[$i]}
    
    echo "Starting fine-tuning for model: $model_name"
    accelerate launch --mixed_precision=bf16 --num_processes=4 --gradient_accumulation_steps=4 train/train_sft_parser_multivarious.py --model_name "$model_name" --save_path "$save_path"
    echo "Finished fine-tuning for model: $model_name"
    echo "======================================="
done


model_paths_w_finetuning=(
    "/path/to/outputs/DeepSeek-7b-tsad-lora"
    "/path/to/outputs/gemma-2-2b-tsad-lora"
    "/path/to/outputs/gemma-7b-tsad-lora"
    "/path/to/outputs/llama-3-8b-tsad-lora"
    "/path/to/outputs/llama-3.2-3B-tsad-lora"
    "/path/to/outputs/phi-4-mini-tsad-lora"
    "/path/to/outputs/qwen2.5-3b-tsad-lora"
    "/path/to/outputs/qwen2.5-7b-tsad-lora"
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
    model_path_w_finetuning="${model_paths_w_finetuning[$i]}"
    model_name="${model_names[$i]}"
    
    echo "==============================================="
    echo "Running model_path: $model_path_w_finetuning | model_name: $model_name"
    echo "==============================================="
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_anomaly_multiclf_multivarious_reason_finetuning.py --model_path "$model_path_w_finetuning" --model_name "$model_name"
    
    echo "Finished running $model_path_w_finetuning with $model_name"
    echo
done
