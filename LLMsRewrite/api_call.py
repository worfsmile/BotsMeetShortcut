# this file call the api to modify text

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from openai import OpenAI
from tqdm import tqdm
import random

api_key = ''


def print_balance():
    url = "https://api.deepseek.com/user/balance"

    payload={}
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)

def print_models():
    # for backward compatibility, you can still use `` as `base_url`.
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    print(client.models.list())

def api_call(messages: str) -> str:
    url = "https://api.deepseek.com/chat/completions"

    payload = json.dumps({
        "messages": messages,
        "model": "deepseek-chat",
        "frequency_penalty": 0,
        "max_tokens": 1024,
        "presence_penalty": 0,
        "response_format": {
            "type": "text"
        },
        "stop": None,
        "stream": False,
        "stream_options": None,
        "temperature": 0.5,
        "top_p": 1,
        "tools": None,
        "tool_choice": "none",
        "logprobs": False,
        "top_logprobs": None
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text




print_balance()



