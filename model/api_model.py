import time
import json
from tqdm import tqdm
import os
import openai
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider

from prompts.prompt import BASE_SYSTEM_PROMPT
from model.hf_model import HFmodel

class APIModel:
    def __init__(self, model_name, api_type, system_prompt=BASE_SYSTEM_PROMPT) -> None:
        self.model_name = model_name
        self.api_type = api_type

        if self.api_type=="azure_openai":
            openai.api_type = "azure"
            openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
            openai.api_version = os.getenv("OPENAI_API_VERSION") 
            openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        elif self.api_type=="openai":
            openai.api_type = "openai"
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api_type=="azure_keyless":
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
            api_version = os.getenv("OPENAI_API_VERSION") 
            self.client = self.azure_keyless(endpoint, api_version=api_version)
        else:
            self.hfmodel = HFmodel(model_name, api_type, system_prompt)

        self.API_RETRY_SLEEP = 10
        self.API_ERROR_OUTPUT = "$ERROR$"
        self.API_QUERY_SLEEP = 10
        self.API_MAX_RETRY = 10
        self.API_TIMEOUT = 20

        self.system_prompt = system_prompt

        
    def generate(self, conv: str,
                max_n_tokens: int, 
                temperature: float,
                top_p: float,
                histories = []):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        messages = []
        messages.append({'role':'system', 'content':self.system_prompt})
        for i in range(len(histories)):
            history_conv, history_ans = histories[i]
            messages.append({'role': 'user', 'content': history_conv})
            messages.append({'role': 'assistant', 'content': history_ans})
        messages.append({'role':'user', 'content':conv})

        output = self.API_ERROR_OUTPUT
        if self.api_type=="azure_openai" or self.api_type=="azure_keyless":
            for _ in range(self.API_MAX_RETRY):
                try:
                    if self.api_type=="azure_openai":
                        response = openai.ChatCompletion.create(
                                    engine = self.model_name,
                                    messages = messages,
                                    max_tokens = max_n_tokens,
                                    temperature = temperature,
                                    top_p = top_p,
                                    request_timeout = self.API_TIMEOUT,
                                    )
                        output = response["choices"][0]["message"]["content"]
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_name,#"gpt-4_1106-Preview",
                            messages=messages,
                            max_tokens=max_n_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        response=json.loads(response.to_json())
                        output = response["choices"][0]["message"]["content"]
                    break
                except openai.OpenAIError as e:
                    print("OpenAIError:", e)
                    time.sleep(self.API_RETRY_SLEEP)
            
                time.sleep(self.API_QUERY_SLEEP)
        elif self.api_type=="openai":
            output = self.openaichat(messages=messages)
        else:
            output = self.hfmodel(conv, histories, max_n_tokens, temperature, top_p)
        return output 
    
    def __call__(self, 
                        conv,
                        histories = [],
                        max_new_tokens: int = 4096, 
                        temperature: float = 1e-8,
                        top_p: float = 1.0):
        assistant = self.generate(conv, max_new_tokens, temperature, top_p, histories)
        histories.append((conv, assistant))
        return assistant, histories

    def azure_keyless(self, endpoint, api_version="2024-08-01-preview"):
        """
        az login --scope https://cognitiveservices.azure.com/.default
        """
        # ChainedTokenCredential example borrowed from
        # https://github.com/technology-and-research/msr-azure/blob/main/knowledge-base/how-to/Access-Storage-Without-Keys-in-Azure-ML.md
        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
                # Azure ML Compute jobs that has the client id of the
                # user-assigned managed identity in it.
                # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
                # In case it is not set the ManagedIdentityCredential will
                # default to using the system-assigned managed identity, if any.
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        return client

    def openaichat(self, messages,top_p=1e-8, temperature=1, max_tokens=4096, **kwargs):
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    top_p=top_p,temperature=temperature,max_tokens=max_tokens,**kwargs
                )
                completion_text=completion.choices[0].message.content
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return completion_text
