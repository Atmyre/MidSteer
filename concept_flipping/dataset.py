import json
import typing as tp
import glob
import random
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from CAA.utils.tokenize import tokenize_llama_chat, tokenize_llama_base


class QuestionsDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 use_chat: bool,
                 device: tp.Any,
                 instruction: str=None):
        # Find all files matching the glob pattern
        matching_files = glob.glob(data_path)
        if not matching_files:
            raise ValueError(f"No files found matching pattern: {data_path}")
            
        # Combine data from all matching files
        self.data = []
        for file_path in matching_files:
            with open(file_path, "r") as f:
                file_data = json.load(f)
                self.data.extend(file_data)
                    
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device
        self.instruction = instruction

    def prompt_to_tokens(self, system_prompt: str | None, user_input: str | None):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                system_prompt=system_prompt,
                user_input=user_input,
            )
        else:
            raise ValueError("Not supported")
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=self.data[idx])


ALPACA_DEFAULT_INSTRUCTION = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"

class AlpacaDataset(QuestionsDataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 use_chat: bool,
                 device: tp.Any,
                 instruction: str=ALPACA_DEFAULT_INSTRUCTION,
                 num_samples: int=None,
                 seed: int=42):
        super().__init__(data_path, tokenizer, use_chat, device, instruction)

        random.seed(seed)
        random.shuffle(self.data)
        if num_samples is not None:
            self.data = self.data[:num_samples]


    def __getitem__(self, idx):
        user_input = self.data[idx]['instruction']
        item_input = self.data[idx]['input']
        if item_input:
            user_input += f"\n\n{item_input}"
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=user_input)
