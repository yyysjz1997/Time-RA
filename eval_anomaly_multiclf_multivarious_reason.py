import argparse
import logging
import os
import json
logger = logging.getLogger(__name__)
import warnings
import re
warnings.filterwarnings("ignore")
from tqdm import tqdm
from model.api_model import APIModel
from prompts.prompt_llama_multi_ano import ANOMALY_DICT, BASE_SYSTEM_PROMPT, USER_DETECTION_PROMPT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metrics.similarity import cos_similarity, tfidf_similarity, levenshtein_similarity, token_sequence_similarity
import numpy as np

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
OPENREVIEW_USERNAME = ""
OPENREVIEW_PASSWORD = ""
LOCAL_HF_PATH = "your_model_path"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for configuring OpenAI API and experiment settings")

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
        "--api_version", type=str, default="2024-02-15-preview", help="API version to be used for making requests. Required for Azure OpenAI clients."
    )
    parser.add_argument(
        "--model_name", type=str, default="llama-3", help="Specifies which LLM model to use."
    )
    parser.add_argument(
        "--data_name", type=str, default="RATs-Multi-TSImage_Reason", help="Dataset name."
    )
    parser.add_argument(
        "--model_path", type=str, default="model_new/Qwen3B-tsad-lora", help="LLM path."
    )
    parser.add_argument(
        "--saving_path", type=str, default="result_multivarious", help="Directory where results, logs, and outputs will be stored."
    )
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="The device to be used for processing (e.g., 'cuda' for GPU acceleration or 'cpu' for standard processing)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path, exist_ok=True)
    print("Running sanity checks for the arguments...")

    if args.openai_client_type == "openai":
        os.environ['OPENAI_API_KEY'] = args.openai_key
        if os.environ.get('OPENAI_API_KEY') is None:
            assert isinstance(args.openai_key, str), ("Please specify the `--openai_key` argument OR set the OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI key is missing.")
    if args.openai_client_type == "azure_openai" or args.openai_client_type == "azure_keyless":
        if args.openai_client_type == "azure_openai" and os.environ.get('AZURE_OPENAI_KEY') is None:
            assert isinstance(args.openai_key, str), ("Please specify the `--openai_key` argument OR set the AZURE_OPENAI_KEY environment variable.")
            os.environ['AZURE_OPENAI_KEY'] = args.openai_key

        if os.environ.get('AZURE_ENDPOINT') is None:
            assert isinstance(args.endpoint, str), ("Please specify the `--endpoint` argument OR set the AZURE_ENDPOINT environment variable.")
            endpoint = args.endpoint
        else:
            endpoint = os.environ.get('AZURE_ENDPOINT')

        if not endpoint.startswith("https://"):
            endpoint = f"https://{endpoint}.openai.azure.com"
        os.environ['AZURE_OPENAI_ENDPOINT'] = endpoint

        if os.environ.get('OPENAI_API_VERSION') is None:
            assert isinstance(args.api_version, str), ("Please specify the `--api_version` argument OR set the OPENAI_API_VERSION environment variable.")
            os.environ['OPENAI_API_VERSION'] = args.api_version

    if args.openai_client_type == "hf_model":   #please enroll your local model
        if os.environ.get('HF_MODEL_PATH') is None:
            # os.environ['HF_MODEL_PATH'] = LOCAL_HF_PATH
            os.environ['HF_MODEL_PATH'] = args.model_path
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
        "Observation": observation,
        "Thought": thought,
        "Action": action
    }

    return output


if __name__ == '__main__':

    args = parse_args()
    api_model = APIModel(args.model_name, args.openai_client_type)

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

    output_json = {}

    if loaded_data is not None:
        merge_json[key] = loaded_data[key]

    for idx in tqdm(data[key].keys()):
        if idx in merge_json[key].keys():
            continue
        best_solution = {}
        observation = data[key][idx]['Observation']
        source = data[key][idx]['Source']
        user_prompt = USER_DETECTION_PROMPT.format(our_observation=observation, our_source=source)
        res, _ = api_model(user_prompt, [])

        action_match = re.search(r"Action:\s*(.*)", res)
        action = action_match.group(1).strip() if action_match else ""
        action = re.sub(r'(?:\\boxed\d+{)?\d*\.?\s*(.*?)}?$', r'\1', action).strip()
        action = re.sub(r'\\boxed\{(.+)$', r"\1", action)

        if action in ANOMALY_DICT:
            anomaly_label = ANOMALY_DICT[action][0]
        else:
            continue

        binary_ID = 0 if anomaly_label == 0 else 1

        output_json[idx] = {
            "Label": binary_ID
        }

        os.makedirs(args.saving_path, exist_ok=True)
        with open(saving_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
