import argparse
import logging
import os
import json
logger = logging.getLogger(__name__)
import warnings
import re
warnings.filterwarnings("ignore")
from tqdm import tqdm
from prompts.prompt_llama_anoclf_reason import ANOMALY_DICT, BASE_SYSTEM_PROMPT, USER_DETECTION_PROMPT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metrics.similarity import cos_similarity, tfidf_similarity, levenshtein_similarity, token_sequence_similarity
from vllm import LLM, SamplingParams
import numpy as np

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
OPENREVIEW_USERNAME = ""
OPENREVIEW_PASSWORD = ""
LOCAL_HF_PATH = "your_model_path"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for configuring OpenAI API and experiment settings")

    # Authentication details for OpenAI API
    parser.add_argument(
        "--openai_key", type=str, default=AZURE_OPENAI_KEY,
        help="API key to authenticate with OpenAI. Can be set via this argument or through the OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--openai_client_type", type=str, default="hf_model", choices=["openai", "azure_openai", "hf_model", "azure_keyless"],
        help="Specify the OpenAI client type to use: 'openai' for standard OpenAI API or 'azure_openai' for Azure-hosted OpenAI services."
    )

    parser.add_argument(
        "--endpoint", type=str, default=AZURE_OPENAI_ENDPOINT,
        help="For Azure OpenAI: custom endpoint to access the API. Should be in the format 'https://<your-endpoint>.openai.azure.com'."
    )

    parser.add_argument(
        "--api_version", type=str, default="2024-02-15-preview", help="API version to be used for making requests. Required "
                                                              "for Azure OpenAI clients."
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="deepseek-7b", choices=["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "llama-3", "qwen-3", "deepseek-7b", "gpt-4_1106-Preview", "gpt-4o_2024-11-20"],
        help="Specifies which LLM model to use."
    )

    parser.add_argument(
        "--data_name", type=str, default="RATs-Multi-TSImage_Reason", help="Dataset name."
    )
    # Output directories
    parser.add_argument(
        "--saving_path", type=str, default="result", help="Directory where results, logs, and outputs will be stored."
    )

    # System configuration
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="The device to be used for processing (e.g., 'cuda' for GPU acceleration or 'cpu' for standard processing)."
    )
    args = parser.parse_args()

    # Ensure necessary directories exist
    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path, exist_ok=True)
    # Sanity checks for authentication
    print("Running sanity checks for the arguments...")

    if args.openai_client_type == "openai":
        os.environ['OPENAI_API_KEY'] = args.openai_key
        if os.environ.get('OPENAI_API_KEY') is None:
            assert isinstance(args.openai_key, str), ("Please specify the `--openai_key` argument OR set the "
                                                      "OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI key is missing.")
    if args.openai_client_type == "azure_openai" or args.openai_client_type == "azure_keyless":
        if args.openai_client_type == "azure_openai" and os.environ.get('AZURE_OPENAI_KEY') is None:
            assert isinstance(args.openai_key, str), ("Please specify the `--openai_key` argument OR set the "
                                                      "AZURE_OPENAI_KEY environment variable.")
            os.environ['AZURE_OPENAI_KEY'] = args.openai_key

        if os.environ.get('AZURE_ENDPOINT') is None:
            assert isinstance(args.endpoint, str), ("Please specify the `--endpoint` argument OR set the "
                                                    "AZURE_ENDPOINT environment variable.")
            endpoint = args.endpoint
        else:
            endpoint = os.environ.get('AZURE_ENDPOINT')

        if not endpoint.startswith("https://"):
            endpoint = f"https://{endpoint}.openai.azure.com"
        os.environ['AZURE_OPENAI_ENDPOINT'] = endpoint

        if os.environ.get('OPENAI_API_VERSION') is None:
            assert isinstance(args.api_version, str), ("Please specify the `--api_version` argument OR set the "
                                                       "OPENAI_API_VERSION environment variable.")
            os.environ['OPENAI_API_VERSION'] = args.api_version

    if args.openai_client_type == "hf_model":   #please enroll your local model
        if os.environ.get('HF_MODEL_PATH') is None:
            os.environ['HF_MODEL_PATH'] = LOCAL_HF_PATH
    print("===================================================================")
    print(args)
    print("===================================================================")

    return args


def parse_to_json(idx, res):

    observation_match = re.search(r"Observation:\s*\[([^\]]+)\]", res)
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:)", res, re.S)
    action_match = re.search(r"Action:\s*\\boxed2\{(.*)\}", res)

    observation = [float(x.strip()) for x in observation_match.group(1).split(",")] if observation_match else []
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""

    output = {
        # str(idx): {
            "Observation": observation,
            "Thought": thought,
            "Action": action
        # }
    }

    return output


if __name__ == '__main__':

    args = parse_args()
    # api_model = APIModel(args.model_name, args.openai_client_type)
    max_model_len = 4096
    model=LLM(LOCAL_HF_PATH, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.8, max_model_len=max_model_len)
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=max_model_len,
                                    top_p= 1.0,
                                    temperature=1e-8,
                                    n=1)
    pred_anotype_list = []
    gt_anotype_list = []

    data_name = args.data_name
    file_path = f"dataset/{data_name}.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    saving_path = f"{args.saving_path}/{data_name}_Reason_by_{args.model_name}.json"

    loaded_data = None
    if os.path.exists(saving_path):
        with open(saving_path, 'r', encoding='utf-8') as file:
            loaded_data = json.load(file)

    merge_json = {}

    key = "TSAD_test"
    merge_json[key] = {}
    print("Deal with the dataset:", key)

    if loaded_data!=None:
        merge_json[key] = loaded_data[key]

    prompt_list = []
    for idx in tqdm(data[key].keys()):
        observation = data[key][idx]['Observation']
        thought_gt = data[key][idx]['Thought']
        source = data[key][idx]['Source']
        action_gt = data[key][idx]['ActionID']
        action_type_gt = data[key][idx]['Action']

        messages = []
        messages.append({'role':'system', 'content':BASE_SYSTEM_PROMPT})
        user_prompt = USER_DETECTION_PROMPT.format(our_observation=observation, our_source=source)
        messages.append({'role': 'user', 'content': user_prompt})
        formatted_messages = tokenizer.apply_chat_template(messages,sampling_params=sampling_params, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_messages, truncation=True, max_length=max_model_len, return_tensors="pt")["input_ids"]
        formatted_messages = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
        print(formatted_messages)
        prompt_list.append(formatted_messages)
    outputs = model.generate(prompt_list, use_tqdm=True,
                                 sampling_params=sampling_params)
    
    for idx, output in enumerate(tqdm(outputs)):
        if idx in merge_json[key].keys():
            continue
        idx = str(idx)
        observation = data[key][idx]['Observation']
        thought_gt = data[key][idx]['Thought']
        source = data[key][idx]['Source']
        action_gt = data[key][idx]['ActionID']
        action_type_gt = data[key][idx]['Action']
        res =  output.outputs[0].text
        action_match = re.search(r"Action:\s*(.*)", res)
        action = action_match.group(1).strip() if action_match else ""
        action = re.sub(r'(?:\\boxed\d+{)?\d*\.?\s*(.*?)}?$', r'\1', action).strip()
        action = re.sub(r'\\boxed\{(.+)$', r"\1", action)
        thought_match = re.search(r"Thought:\s*(.*)", res)
        thought = thought_match.group(1).strip() if thought_match else ""

        if action in ANOMALY_DICT:
            anomaly_label = ANOMALY_DICT[action][0]
        else:
            print(f"\nSkipping - Action '{action}' not found in ANOMALY_DICT")
            continue

        if not observation or not action:
            print("\nSkipping - Missing observation, thought or action")
            continue

        pred_anotype_list.append(anomaly_label)
        gt_anotype_list.append(action_gt)
        print("Type Accuracy:", accuracy_score(gt_anotype_list, pred_anotype_list))

        pred_ano_list = [1 if x != 0 else 0 for x in pred_anotype_list]
        gt_ano_list = [1 if x != 0 else 0 for x in gt_anotype_list]
        print("Binary Accuracy:", accuracy_score(gt_ano_list, pred_ano_list))

        proposed_solutions = [thought, thought_gt]
        print("Thought",thought)
        print("Thought_GT",thought_gt)
        best_solution = {}
        best_solution['cosine_similarity'] = float(cos_similarity(proposed_solutions))
        best_solution['tfidf_similarity'] = tfidf_similarity(proposed_solutions)
        best_solution['levenshtein_similarity'] = levenshtein_similarity(proposed_solutions)
        best_solution['token_sequence_similarity'] = token_sequence_similarity(proposed_solutions)

        print('Reasoning/Thought Similarity:', best_solution)


        merge_json[str(key)][str(idx)] = {
                "Observation": observation,
                "Thought": thought,
                "Action": action,
                "ActionID": anomaly_label,
                "thought_similarity": best_solution
        }

        with open(saving_path, 'w') as f:
            json.dump(merge_json, f, indent=4)

    print("Type Precision:", precision_score(gt_anotype_list, pred_anotype_list, average='macro', zero_division=0))
    print("Type Recall:", recall_score(gt_anotype_list, pred_anotype_list, average='macro', zero_division=0))
    print("Type F1-score:", f1_score(gt_anotype_list, pred_anotype_list, average='macro', zero_division=0))
    print("Binary Precision:", precision_score(gt_ano_list, pred_ano_list, average='macro', zero_division=0))
    print("Binary Recall:", recall_score(gt_ano_list, pred_ano_list, average='macro', zero_division=0))
    print("Binary F1-score:", f1_score(gt_ano_list, pred_ano_list, average='macro', zero_division=0))
    np.save(os.join(args.saving_path, args.model_name,'pred_anotype_list.npy'), pred_anotype_list)
    np.save(os.join(args.saving_path, args.model_name,'gt_anotype_list.npy'), gt_anotype_list)

            