#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_names=(
    "/path/to/model/deepseek-llm-7b-chat"
    "/path/to/model/gemma-2-2b-it"
    "/path/to/model/gemma-7b-it"
    "/path/to/model/llama-3-8b-instruct"
    "/path/to/model/llama-3.2-3B-Instruct"
    "/path/to/model/phi-4-mini-instruct"
    "/path/to/model/qwen2.5-3b-instruct"
    "/path/to/model/qwen2.5-7b-instruct"
)

save_paths=(
    "model_new/DeepSeek-7b-tsad-lora"
    "model_new/gemma-2-2b-tsad-lora"
    "model_new/gemma-7b-tsad-lora"
    "model_new/llama-3-8b-tsad-lora"
    "model_new/llama-3.2-3B-tsad-lora"
    "model_new/phi-4-mini-tsad-lora"
    "model_new/qwen2.5-3b-tsad-lora"
    "model_new/qwen2.5-7b-tsad-lora"
)

for ((i=0; i<${#model_names[@]}; i++))
do
    model_name=${model_names[$i]}
    save_path=${save_paths[$i]}
    
    echo "Starting fine-tuning for model: $model_name"
    accelerate launch --mixed_precision=bf16 --num_processes=4 --gradient_accumulation_steps=4 train/train_sft_parser.py --model_name "$model_name" --save_path "$save_path"
    echo "Finished fine-tuning for model: $model_name"
    echo "======================================="
done

model_paths_w_finetuning=(
    "/path/to/model_new/DeepSeek-7b-tsad-lora"
    "/path/to/model_new/gemma-2-2b-tsad-lora"
    "/path/to/model_new/gemma-7b-tsad-lora"
    "/path/to/model_new/llama-3-8b-tsad-lora"
    "/path/to/model_new/llama-3.2-3B-tsad-lora"
    "/path/to/model_new/phi-4-mini-tsad-lora"
    "/path/to/model_new/qwen2.5-3b-tsad-lora"
    "/path/to/model_new/qwen2.5-7b-tsad-lora"
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
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_anomaly_multiclf_reason_finetuning.py --model_path "$model_path_w_finetuning" --model_name "$model_name"
    
    echo "Finished running $model_path_w_finetuning with $model_name"
    echo
done
