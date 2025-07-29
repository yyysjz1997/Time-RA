import os
import json
import urllib.request
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
import re
import json
import sys
from pathlib import Path

# Path anonymized
sys.path.append('/path/to/project/Time-RA')
from prompts.prompt_llama_multi_ano import USER_DETECTION_LABELED_PROMPT

def split_ranking(s):
    return re.findall(r'[^>=]+', s)

scope = "api://trapi/.default"
credential = get_bearer_token_provider(AzureCliCredential(), scope)

deployment_name = "gpt-4o"
endpoint = ""
model_version = "2024-05-13"
deployment_name = f"{deployment_name}_{model_version}"
api_version = '2024-09-01-preview'
client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)

# Path needs to change
model_pool_path = {
    "gpt-4o":"/path/to/project/Time-RA/dataset/reason_data/TSAD-Reasoning-TSImage_new_Reason_by_gpt-4o_2024-11-20.json",
    "llama-70b":"/path/to/project/Time-RA/labeling/dataset/reason_data/TSAD-Reasoning-TSImage_Reason_by_llama-70b.json",
    "deepseek-r1":"/path/to/project/Time-RA/labeling/dataset/reason_data/TSAD-Reasoning-TSImage_Reason_by_deepseek-r1.json",
    "gemini":"/path/to/project/Time-RA/labeling/dataset/reason_data/TSAD-Reasoning-TSImage_Reason_by_gemini.json",
}
save_path = '/path/to/project/Time-RA/dataset/reason_data/TSAD-Reasoning-TSImage-by-gpt4-ranking.json'

with open(model_pool_path["gpt-4o"], 'r', encoding='utf-8') as f:
    gpt_model = json.load(f)
    num = len(gpt_model["TSAD_test"].keys())
print("instance number is", num)
file_path = Path(save_path)
if file_path.exists():
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    results = {"TSAD_test": {}}
    for i in range(num):
        results["TSAD_test"][str(i)]=None

model_pool = {}
for key in model_pool_path.keys():
    saving_path = model_pool_path[key]
    if os.path.exists(saving_path):
        with open(saving_path, 'r', encoding='utf-8') as file:
            loaded_data = json.load(file)["TSAD_test"]
            model_pool[key] = loaded_data

for i in range(num):
    if results["TSAD_test"][str(i)]!=None:
        continue
    if str(i) not in model_pool["gpt-4o"].keys():
        continue
    joint_prompt = "Now we have the outputs of models, there are:\n\n"
    model_str = ""
    for key in model_pool.keys():
        model = model_pool[key]
        model_str += key+", "
        if "gpt-4o"==key:
            obs = model[str(i)]["Observation"]
            label = model[str(i)]["Label"]
            source = model[str(i)]["Source"]
            prompt = USER_DETECTION_LABELED_PROMPT.format(our_label=label, our_source=source, our_observation=obs)
            joint_prompt = "A task we have is:\n【"+prompt+"】\n"+joint_prompt
        if str(i) in model.keys():
            cur_prompt = f"The model 【{key}】 output is:\n"
            thought = model[str(i)]["Thought"]
            action = model[str(i)]["Action"]
            cur_prompt+=f"Thought: {thought}\nAction: {action}"
            joint_prompt+=f"{cur_prompt}\n\n"
    joint_prompt+=f"Please provide me with a ranking to compare the output results of these models from {model_str}, here is an example of the output format:\n"
    joint_prompt+="<|begin|>gpt-4o>gemini>deepseek-r1>llama-70b<|end|>"

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": joint_prompt,
            },
        ]
    )
    resp = response.choices[0].message.content
    resp = resp.split("<|begin|>")[-1].split("<|end|>")[0].strip()
    try:
        split_rank = split_ranking(resp)
        print(i, split_rank)
        results["TSAD_test"][str(i)] = model_pool[split_rank[0]][str(i)]
        results["TSAD_test"][str(i)]["Rank"] = split_rank
    except Exception as e:
        results["TSAD_test"][str(i)] = model_pool["gpt-4o"][str(i)]
        results["TSAD_test"][str(i)]["Rank"] = ["gpt-4o", "deepseek-r1", "gemini", "llama-70b"]
    
    with open(save_path, 'w') as fp:
        json.dump(results, fp)