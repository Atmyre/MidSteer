import json
import typing as tp

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers import AutoTokenizer

from CAA.utils.tokenize import tokenize_llama_chat, tokenize_llama_base
from CAA.generate_vectors import ComparisonDataset

    
class AlpacaDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 use_chat: bool,
                 device: tp.Any,
                 *,
                 pos_concept: str | None = None,
                 neg_concept: str | None = None):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device
        self._pos_concept = pos_concept
        self._neg_concept = neg_concept

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
        p_inst_text = inst_text
        if self._pos_concept is not None:
            p_inst_text += ' ' + self._pos_concept
        n_inst_text = inst_text
        if self._neg_concept is not None:
            n_inst_text += ' ' + self._neg_concept
        p_tokens = self.prompt_to_tokens(p_inst_text, input_text)
        n_tokens = self.prompt_to_tokens(n_inst_text, input_text)
        return p_tokens, n_tokens


class PairedQuestionsDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 use_chat: bool,
                 device: tp.Any,
                 *,
                 pos_concept: str = None,
                 neg_concept: str = None):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device
        self._pos_concept = pos_concept
        self._neg_concept = neg_concept

    def prompt_to_tokens(self, system_prompt: str | None, instruction: str | None):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                system_prompt=system_prompt,
                user_input=instruction,
            )
        else:
            raise ValueError("Not supported")
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos_question = self.data[self._pos_concept][idx]
        neg_question = self.data[self._neg_concept][idx]
        p_inst_text = ''#f'Answer the following question about {self._pos_concept}.'
        n_inst_text = ''#f'Answer the following question about {self._neg_concept}.'
        p_tokens = self.prompt_to_tokens(p_inst_text, pos_question)
        n_tokens = self.prompt_to_tokens(n_inst_text, neg_question)
        return p_tokens, n_tokens




def init_model_and_tokenizer(model_name: str, cache_dir: str | None = './cache') -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if '3.1' in model_name:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    # ***REMOVED***
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map='balanced',
        token='***REMOVED***'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map='balanced',
        token='***REMOVED***'
    )
    return model, tokenizer
