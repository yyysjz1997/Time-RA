import argparse
import logging
import os
import json
logger = logging.getLogger(__name__)
import warnings
import re
warnings.filterwarnings("ignore")
from tqdm import tqdm
import sys

PROJECT_ROOT = '/path/to/project/root'
HUGGINGFACE_PATH = '/path/to/huggingface'
sys.path.append(f'{PROJECT_ROOT}/Time-RA')
import base64

from model.api_model import APIModel
from prompts.prompt_llama_multi_ano import *
AZURE_OPENAI_ENDPOINT = "" 
AZURE_OPENAI_KEY = ""
OPENREVIEW_USERNAME = ""
OPENREVIEW_PASSWORD = ""
LOCAL_HF_PATH = "/path/to/huggingface/Llama-3.1-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for configuring OpenAI API and experiment settings")

    # Authentication details for OpenAI API
    parser.add_argument(
        "--openai_key", type=str, default=AZURE_OPENAI_KEY,
        help="API key to authenticate with OpenAI. Can be set via this argument or through the OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--openai_client_type", type=str, default="azure_keyless", choices=["openai", "azure_openai", "hf_model", "azure_keyless", "gemini", "deepseek-r1", "llama-70b"],
        help="Specify the OpenAI client type to use: 'openai' for standard OpenAI API or 'azure_openai' for Azure-hosted OpenAI services."
    )

    parser.add_argument(
        "--endpoint", type=str, default=AZURE_OPENAI_ENDPOINT,
        help="For Azure OpenAI: custom endpoint to access the API. Should be in the format 'https://<your-endpoint>.openai.azure.com'."
    )

    parser.add_argument(
        "--api_version", type=str, default="2024-05-01-preview", help="API version to be used for making requests. Required "
                                                              "for Azure OpenAI clients."
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o_2024-11-20",
        help="Specifies which LLM model to use."
    )
    parser.add_argument(
        "--model_path", type=str, default="gpt-4o_2024-11-20",
    )

    parser.add_argument(
        "--data_name", type=str, default="RATs-Multi-TSImage_Reason", help="Dataset name."
    )
    # Output directories
    parser.add_argument(
        "--saving_path", type=str, default="dataset/reason_data", help="Directory where results, logs, and outputs will be stored."
    )

    # System configuration
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="The device to be used for processing (e.g., 'cuda' for GPU acceleration or 'cpu' for standard processing)."
    )
    parser.add_argument("--is_obs", action="store_true")

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
        os.environ['HF_MODEL_PATH'] = args.model_path
    print("===================================================================")
    print(args)
    print("===================================================================")

    return args

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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
    api_model = APIModel(args.model_name, args.openai_client_type)
    data_name = args.data_name

    is_img = False
    is_obs = args.is_obs

    base_path = f"{HUGGINGFACE_PATH}/Time-RA/"
    file_path = os.path.join(base_path, f"{data_name}.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    saving_path = os.path.join(args.saving_path, f"{data_name}_Reason_by_{args.model_name}.json")
    if is_img:
        if not is_obs:
            saving_path = os.path.join(args.saving_path, f"{data_name}_Reason_by_{args.model_name}_img_wo_obs.json")
        else:
            saving_path = os.path.join(args.saving_path, f"{data_name}_Reason_by_{args.model_name}_img_with_obs.json")

    # saving_path = os.path.join(args.saving_path, f"{data_name}_Reason_by_gpt-4o_2024-11-20.json")
    print(saving_path)
    loaded_data = None
    if os.path.exists(saving_path):
        with open(saving_path, 'r', encoding='utf-8') as file:
            loaded_data = json.load(file)

    merge_json = {}
    num = 0
    for key in data.keys():
        if key != "TSAD_test":
            continue

        print("Deal with the dataset:", key)
        if loaded_data is not None and key in loaded_data.keys():
            merge_json[key] = loaded_data[key]
            num += len(merge_json[key].keys())
            print(key, len(merge_json[key].keys()), len(data[key].keys()))
        else:
            merge_json[key] = {}

    for key in data.keys():
        if key != "TSAD_test":
            continue
        for idx in tqdm(data[key].keys()):
            if idx in merge_json[key].keys():
                continue

            observation = data[key][idx]['Time_series']
            label = data[key][idx]['Label']
            source = data[key][idx]['Source']
            figurepath = data[key][idx]['FigurePath'].replace(".pdf", ".jpg")
            if is_img:
                if not is_obs:
                    observation = "Please refer to the attached picture."

                user_prompt = USER_DETECTION_LABELED_PROMPT.format(our_label=label, our_source=source, our_observation=observation)
                if "llava" in args.model_name:
                    user_prompt = USER_DETECTION_PROMPT_IMG_LLAVA.format(our_label=label, our_source=source, our_observation=observation)
                elif "gemma" in args.model_name:
                    user_prompt = USER_DETECTION_PROMPT_IMG_GEMMA.format(our_label=label, our_source=source, our_observation=observation)
                res, _ = api_model(user_prompt, [], img=os.path.join(base_path, figurepath))
            else:
                user_prompt = USER_DETECTION_LABELED_PROMPT.format(our_label=label, our_source=source, our_observation=observation)
                res, _ = api_model(user_prompt, [])

            thought_match = re.search(r"Thought:\s*(.*?)\nAction", res, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*)", res)

            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip() if action_match else ""

            action = re.sub(r'(?:\\boxed\d+{)?\d*\.?\s*(.*?)}?$', r'\1', action).strip()
            action = action.strip().replace("anomaly", "Anomaly")
            if re.search(r'\\boxed\d?\{.*?\}', thought):
                thought = re.sub(r'\\boxed\d?\{(.*?)\}', r'\1', thought).strip()
            if re.search(r'\\boxed\d?\{.*?\}', thought):
                thought = re.sub(r'\\boxed\d?\{(.*?)\}', r'\1', thought).strip()
            # print(action)
            # print("="*30)
            if action in ANOMALY_DICT:
                anomaly_lable = ANOMALY_DICT[action][0]
            else:
                print(f"\nSkipping - Action '{action}' not found in ANOMALY_DICT")
                print(res)
                continue
            if not observation or not thought or not action:
                print("\nSkipping - Missing observation, thought or action")
                continue

            merge_json[str(key)][str(idx)] = {
                    "Observation": observation,
                    "Source": source,
                    "FigurePath": figurepath,
                    "Thought": thought,
                    "Action": action,
                    "ActionID": anomaly_lable,
                    "Label": label,
            }

            with open(saving_path, 'w') as f:
                json.dump(merge_json, f, indent=4)
    print(num)