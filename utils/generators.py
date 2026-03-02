import base64
from PIL import Image
from io import BytesIO
import requests
import time
import abc
import numpy as np

from dotenv import load_dotenv
load_dotenv()

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_ROUTER_CONFIG = {"order": ["Lambda", "DeepInfra", "Kluster", "InferenceNet"], "require_parameters": True, "quantizations": ["bf16"]}

SHORT_MODEL_NAMES = {
    #VLM
    "openai/gpt-4o-2024-08-06": "gpt4o",
    "meta-llama/llama-3.2-90b-vision-instruct": "llama3.2_90b",
    #LLM
    "meta-llama/llama-3.3-70b-instruct": "llama3.3_70b",
    "google/gemma-3-12b-it": "gemma3_12b",
    'openai/gpt-4o-mini': "gpt4o_m"
}

proxies = None

class LanguageModel(abc.ABC):
    """Abstract base class for all language models."""
    def __init__(self, model_name):
        self.model_name = model_name

    @abc.abstractmethod
    def conversation_answer(self, conversation: list[dict]) -> tuple[str, int]:
        """Generate a next message from model.
        Returns answer from LLM and number of processed tokens.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def message_answer(self, message: str, image : np.array =None) -> tuple[str, int]:
        """Generate a answer on message from LM model.
        Returns answer from LLM and number of processed tokens.
        """
        raise NotImplementedError

class OpenRouterModel(LanguageModel):
    def __init__(self, model_name, api_key, provider_config = None, short_name = None):
        super().__init__(model_name)
        self.header = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        self._set_config(provider_config)
        self.short_name = model_name
        if short_name:
            self.short_name = short_name
        if self.short_name in SHORT_MODEL_NAMES:
            self.short_name =  SHORT_MODEL_NAMES[model_name]
    
    def _encode_image(self, image) -> str:
        """Encodes an numpy RGB image array to a base64 string."""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

    def _set_config(self, config):
        if config:
            self.provider_config = config
            return
        else:
            self.provider_config = {}
        if "openai" not in self.model_name:
            self.provider_config = DEFAULT_ROUTER_CONFIG
            return
        
    def _get_response(self, parameters):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=self.header,
                timeout=5,
                json=parameters,
                proxies = proxies
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}. Try again.")
            try:
                print("Response JSON:", response.json())
            except Exception as e:  # корректный тип исключения для json()
                print("No response JSON.")
            print("Trying again.")
            time.sleep(3)
            return None

        answer = response.json()
        if not (answer.get("choices") and 
                answer["choices"][0].get("message") and 
                answer["choices"][0]["message"].get("content")):
            raise ValueError(f"\nInvalid response: {answer}")
        return response

    def conversation_answer(self, conversation):
        parameters={
                "model": self.model_name,
                "messages": conversation,
                "temperature": DEFAULT_TEMPERATURE,
                "seed": DEFAULT_SEED,
                #"provider": {"order": [ "Azure"]},
            }
        for _ in range(10):
            response = self._get_response(parameters)
            if not response:
                continue
            completion = response.json()
            return completion['choices'][0]['message']['content'], completion['usage']['prompt_tokens']
        raise ValueError(f"Too many errors during OpenRouter API request. Shutting down...")
    
    def message_answer(self, message, image=None):
        content = [{"type": "text", "text": message}]
        if image is not None:
            base64_image = self._encode_image(image)
            data_url = f"data:image/jpeg;base64,{base64_image}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        parameters={
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": DEFAULT_TEMPERATURE,
            "seed": DEFAULT_SEED,
            #"provider": {"order": [ "Azure"]}
        }

        for _ in range(10):
            response = self._get_response(parameters)
            if not response:
                continue
            completion = response.json()
            return completion['choices'][0]['message']['content'], completion['usage']['prompt_tokens']
        raise ValueError(f"Too many errors during OpenRouter API request. Shutting down...")

class OllamaModel(LanguageModel):
    """LLM implementation using a local Ollama instance."""
    ollama_model_map = {
        #LLM
        "meta-llama/llama-3.3-70b-instruct": "llama3.3:70b",
        "meta-llama/llama-3.1-8b-instruct": "llama3.1:8b",
        "google/gemma-3-12b-it": "gemma3:12b",
        #VLM
        "meta-llama/llama-3.2-90b-vision-instruct": "llama3.2-vision:90b"
    }

    def __init__(self, model_name, server_url, options = None):
        super().__init__(self.ollama_model_map.get(model_name))
        if not self.model_name:
            raise ValueError(f"Model {model_name} not found in OpenRouter to Ollama model map")
        self.url = server_url + "/api/chat"
        self.options = {"temperature": DEFAULT_TEMPERATURE, "seed": DEFAULT_SEED}
        if options:
            self.options = options

    def _encode_image(self, image) -> str:
        """Encodes an numpy RGB image array to a base64 string."""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def _get_responce(self, payload):
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        answer = response.json()
        if not (answer.get("eval_count") and 
        answer.get("prompt_eval_count")):
            raise ValueError(f"Invalid response: {answer}")
        tokens = answer['eval_count'] + answer['prompt_eval_count']
        return answer['message']['content'], tokens
            
    def conversation_answer(self, conversation: list) -> tuple[str | None, int | None]:
        payload = {
            "model": self.model_name,
            "messages": conversation,
            "options": self.options,
            "stream": False
        }
        return self._get_responce(payload)
    
    def message_answer(self, message, image = None):
        base64_image = self._encode_image(image)
        payload = {
            "model": self.model_name,
            "prompt": message,
            "stream": False,
            "options": self.options,
            "images": [base64_image]
        }
        return self._get_responce(payload)
