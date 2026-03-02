import yaml
import torch
import requests
import time
# GPT
import openai
import os

# HuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = None
model = None
config = None


def load_config(path="amb_detector/llm_config.yaml"):
    global config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_model():
    global tokenizer, model, config

    if config is None:
        load_config()

    model_type = config["model"]["type"]

    if model_type == "mistral":
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["path"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    elif model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-7B-Chat",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    elif model_type == "gpt":
        openai.api_key = config["model"]["api_key"]  
        model = "gpt-4-turbo"

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def llm_response(system_prompt, user_prompt, max_tokens=500):
    global tokenizer, model, config

    if config is None or model is None:
        load_config()
        init_model()

    model_type = config["model"]["type"]

    if model_type == "mistral":
        prompt = f"<s>[INST] {system_prompt.strip()}\n\n{user_prompt.strip()} [/INST]"
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        result = decoded.split("[/INST]", 1)[1].strip() if "[/INST]" in decoded else decoded.strip()
        return result

    elif model_type == "qwen":
        prompt = (
            f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        return decoded.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()

    elif model_type == "gpt":
        prompt = user_prompt
        conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        key = os.getenv("OPEN_ROUTER_KEY")
        #model = "openai/gpt-4.1"
        model = "openai/gpt-3.5-turbo"
        answer, tokens = conversation_answer(conv,model, key)
        return answer.strip()
        # response = openai.ChatCompletion.create(
        #     model="gpt-4-turbo",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=config["model"].get("temperature", 0.0),
        #     max_tokens=max_tokens
        # )
        # return response["choices"][0]["message"]["content"].strip()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

proxie = None

def _get_response(parameters, api_key):
    header = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=header,
            timeout=5,
            json=parameters,
            proxies=proxie
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}. \nTry again.")
        time.sleep(3)
        return None

    answer = response.json()
    if not (answer.get("choices") and 
            answer["choices"][0].get("message") and 
            answer["choices"][0]["message"].get("content")):
        raise ValueError(f"\nInvalid response: {answer}")
    return response

def conversation_answer(conversation, model, api_key):
    parameters={
            "model": model,
            "messages": conversation,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": 500
        }
    for _ in range(10):
        response = _get_response(parameters, api_key)
        if not response:
            continue
        completion = response.json()
        return completion['choices'][0]['message']['content'], completion['usage']['prompt_tokens']
    raise ValueError(f"Too many errors during OpenRouter API request. Shutting down...")