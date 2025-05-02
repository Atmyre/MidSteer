import json
import typing as tp

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers import AutoTokenizer


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str | None,
    model_output: str = None,
    system_prompt: str = None,
) -> list[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    if user_input is not None:
        input_content += f"{B_INST} {user_input.strip()} {E_INST} "
    if model_output is not None:
        input_content += f"{model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> list[int]:
    input_content = ""
    if user_input is not None:
        input_content += f"{BASE_INPUT} {user_input.strip()} {BASE_RESPONSE} "
    if model_output is not None:
        input_content += f"{model_output.strip()}"
    return tokenizer.encode(input_content)


class ComparisonDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, use_chat: bool, device: tp.Any):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device

    def prompt_to_tokens(self, instruction: str | None, model_output: str):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item.get("question")
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens
    
class AlpacaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, use_chat: bool, device: tp.Any):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device

    def prompt_to_tokens(self, system_prompt: str | None, instruction: str | None):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                system_prompt=system_prompt,
                user_input=instruction,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                system_prompt=system_prompt,
                user_input=instruction,
            )
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inst_text = item["instruction"]
        input_text = item["input"]
        inst_text1 = inst_text+' '+'Think about horses when you answer the question.'
        inst_text2 = inst_text+' '+'Think about motorbikes when you answer the question.'
        p_tokens = self.prompt_to_tokens(inst_text1, input_text)
        n_tokens = self.prompt_to_tokens(inst_text2, input_text)
        return p_tokens, n_tokens


def init_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # ***REMOVED***
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir='./cache',
        torch_dtype=torch.float16,
        device_map='balanced',
        token='***REMOVED***'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='./cache',
        torch_dtype=torch.float16,
        device_map='balanced',
        token='***REMOVED***'
    )
    return model, tokenizer
