from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import os
import transformers
import torch

from prompts.prompt import BASE_SYSTEM_PROMPT

class HFmodel:
    def __init__(self, model_name="llama-3", api_type="hf_model", system_prompt=BASE_SYSTEM_PROMPT):

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=os.getenv("HF_MODEL_PATH"),
            model_kwargs={"torch_dtype": torch.bfloat16},
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model_name = model_name
        self.system_prompt = system_prompt

    def prepare_prompt(self, conv:str, histories=[]):
        messages = []
        messages.append({'role':'system', 'content':self.system_prompt})
        for i in range(len(histories)):
            history_conv, history_ans = histories[i]
            messages.append({'role': 'user', 'content': history_conv})
            messages.append({'role': 'assistant', 'content': history_ans})
        messages.append({'role':'user', 'content':conv})

        return messages
    
    def __call__(self, 
                        conv,
                        histories = [],
                        max_new_tokens: int = 4096, 
                        temperature: float = 1e-8,
                        top_p: float = 1.0,
                        n=1):
        if temperature < 0.0001:
            temperature = 0.0001
        msgs = self.prepare_prompt(conv)
        msgs = [msg for msg in msgs if msg["role"] != "system"] # gemma
        
        # print("MSGS\n",msgs)
        outputs = self.pipeline(
            msgs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pipeline.tokenizer.eos_token_id, 
            num_return_sequences=n,
            )
        outs = outputs[0]["generated_text"][-1]["content"]
        # print(outs)
        return outs
