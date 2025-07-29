import json
import torch
import math
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    default_data_collator,
    get_scheduler
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import os
import traceback
import argparse

def get_dynamic_batch_size(base_batch_size=4):
    # Dynamically adjust batch size based on available GPU memory
    free_mem = torch.cuda.mem_get_info()[0]
    free_mem_in_GB = free_mem / (1024 ** 3)

    if free_mem_in_GB > 40:
        scale = 2
    elif free_mem_in_GB > 20:
        scale = 1
    elif free_mem_in_GB > 10:
        scale = 0.5
    else:
        scale = 0.25

    adjusted_batch_size = max(1, int(base_batch_size * scale))
    print(f"Adjusted batch size: {adjusted_batch_size} based on free memory: {free_mem_in_GB:.2f} GB")
    return adjusted_batch_size

def main(model_name, save_path):
    # === Step 1: Load dataset ===
    with open("path/RATs-Multi-TSImage_Reason.json") as f:
        raw_data = json.load(f)["TSAD_train"]

    formatted_data = []
    for key, entry in raw_data.items():
        formatted_data.append({
            "observation": entry["Observation"],
            "thought": entry["Thought"],
            "action": entry["Action"],
            "source": entry["Source"]
        })

    dataset = Dataset.from_list(formatted_data).train_test_split(test_size=0.2, shuffle=True, seed=42)

    # === Step 2: Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        observations = [", ".join(map(str, obs)) for obs in examples["observation"]]
        sources = examples["source"]
        thoughts = examples["thought"]
        actions = examples["action"]

        prompts = [
            f"You are an expert in time series anomaly detection. We provide a time series (called Observation), you should give us the anomaly type (called Action) and its reasons (called Thought). \n"
            f"Thought steps can infer the current abnormal situation of time series.\n\n"
            f"The anomaly detection of each time series is divided into three steps: Observation, Thought, Action. After analyzing each observation, please provide the next Thought and next Action.\n"
            f"Here is a time series observation that we need to check for anomaly categories. The oberservation is from domain of {s}.\n"
            f"Please make a Thought judgment and put your final Action within \\boxed1{{}} and \\boxed2{{}} respectively, where action must just be a category name not id.\n"
            f"Observation: {o}\n"
            f"Thought: \\boxed1{{}}\n"
            f"Action: \\boxed2{{}}\n"
            for o, s in zip(observations, sources)
        ]

        targets = [f"Thought: \\boxed1{{{t}}}\nAction: \\boxed2{{{a}}}" for t, a in zip(thoughts, actions)]

        full_texts = []
        for prompt, target in zip(prompts, targets):
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]
            full_text = tokenizer.apply_chat_template(chat, tokenize=False)
            full_texts.append(full_text)

        tokenized = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # === Step 4: Dataloaders ===

    dynamic_batch_size = get_dynamic_batch_size(base_batch_size=4)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=dynamic_batch_size,
        collate_fn=default_data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["test"],
        batch_size=dynamic_batch_size,
        collate_fn=default_data_collator
    )

    # === Step 5: Load model and wrap with LoRA ===
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_cache = False
    config.use_flash_attention_2 = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # === Step 6: Setup accelerate ===
    accelerator = Accelerator()
    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    # === Step 7: Optimizer & Scheduler ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * 3,
    )

    # === Step 8: Training loop ===
    model.train()
    for epoch in range(1):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})

        accelerator.print(f"Epoch {epoch+1} finished, avg loss: {total_loss / len(train_dataloader):.4f}")

    # === Step 9: Save model ===
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        try:
            accelerator.unwrap_model(model).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        except Exception:
            print("Error occurred while saving the model:")
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a LLM with LoRA on TSAD reasoning data.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model.")
    args = parser.parse_args()

    main(args.model_name, args.save_path)
