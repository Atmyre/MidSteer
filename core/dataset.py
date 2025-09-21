import json
import typing as tp
import glob
import random
import torch
import re
import itertools
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets import load_dataset

from CAA.utils.tokenize import tokenize_llama_chat


def dumb_tokenizer_fn(
        tokenizer: AutoTokenizer | None,
        user_input: str,
        system_prompt: str | None = None,
        model_output: str | None = None,
) -> str:
    parts = [p for p in [system_prompt, user_input, model_output] if p is not None]
    return " ".join(parts)


def tokenize_with_chat_template(
        tokenizer: AutoTokenizer,
        user_input: str,
        system_prompt: str | None = None,
        model_output: str | None = None,
) -> torch.Tensor:
    if model_output is not None:
        raise ValueError("Model output is not supported for chat template tokenizer function")
    conversation = []
    if system_prompt is not None:
        conversation.append({
            "role": "system",
            "content": system_prompt
        })
    conversation.append({
        "role": "user",
        "content": user_input
    })
    return tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )

def resolve_tokenizer_for_model(model_name: str) -> tp.Callable:
    if 'llama' in model_name.lower() and 'chat' in model_name.lower():
        return tokenize_llama_chat  # This is needed to obtain correct steering vectors for LLAMA
    else:
        return tokenize_with_chat_template

class QuestionsDataset(Dataset):
    def __init__(self,
                 *,
                 data_path: str | None = None,
                 data: tp.Iterable[tp.Any] | None = None,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = tokenize_llama_chat,
                 device: tp.Any = None,
                 instruction: str | None = None,
                 dataset_slice: slice | None = None,
                 seed: int | None = None):
        if (data_path is not None) ^ (data is not None) == False:
            raise ValueError("Exactly one of data_path or data should be provided")
        if data_path is not None:
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
        else:
            self.data = list(data)

        if seed is not None:
            random.seed(seed)
            random.shuffle(self.data)
        if dataset_slice is not None:
            self.data = self.data[dataset_slice]
                    
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        if self.tokenizer is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.instruction = instruction

    def prompt_to_tokens(self, system_prompt: str | None, user_input: str | None, model_output: str | None = None):
        tokens = self.tokenizer_fn(
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_input=user_input,
            model_output=model_output,
        )
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=self.data[idx])


ALPACA_DEFAULT_INSTRUCTION = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"

class AlpacaDataset(QuestionsDataset):
    def __init__(self,
                 data_path: str,
                 *,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = tokenize_llama_chat,
                 device: tp.Any = None,
                 instruction: str | None = ALPACA_DEFAULT_INSTRUCTION,
                 dataset_slice: slice | None = None,
                 seed: int | None = None,
                 include_model_output: bool = False):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            tokenizer_fn=tokenizer_fn,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
            seed=seed,
        )
        self.include_model_output = include_model_output


    def __getitem__(self, idx):
        user_input = self.data[idx]['instruction']
        item_input = self.data[idx]['input']
        if self.include_model_output:
            model_output = self.data[idx]['output']
        else:
            model_output = None
        if item_input:
            user_input += f"\n\n{item_input}"
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=user_input, model_output=model_output)


class TemplateDataset(QuestionsDataset):
    def __init__(self,
                 template_path: str,
                 concept: str,
                 *,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = tokenize_llama_chat,
                 device: tp.Any = None,
                 instruction: str | None = None,
                 dataset_slice: slice | None = None,
                 seed: int | None = None):
        super().__init__(
            data_path=template_path,
            tokenizer=tokenizer,
            tokenizer_fn=tokenizer_fn,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
            seed=seed,
        )
        self.concept = concept

    def __getitem__(self, idx):
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=self.data[idx].format(self.concept))


MMLU_SYSTEM_PROMPT = "Elaborate on the following user question."

class MMLUDataset(QuestionsDataset):
    def __init__(self,
                 data_path: str,
                 *,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = tokenize_llama_chat,
                 device: tp.Any = None,
                 instruction: str | None = None,
                 dataset_slice: slice | None = None,
                 seed: int | None = None):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            tokenizer_fn=tokenizer_fn,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
            seed=seed,
        )

    def __getitem__(self, idx):
        full_prompt = self.data[idx]['prompt']
        return self.prompt_to_tokens(system_prompt=self.instruction, user_input=full_prompt)

RELAION_NUM_FILES = 1

class RelaionDataset(QuestionsDataset):
    """Dataset that loads Re-LAION captions and filters them based on a concept."""
    
    def __init__(self,
                 *,
                 concept: str | None = None,
                 max_samples: int | None = None,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = dumb_tokenizer_fn,
                 device: tp.Any = None,
                 instruction: str | None = None,
                 dataset_slice: slice | None = None,
                 seed: int | None = None):
        
        def generator():
            """Load Re-LAION dataset from HuggingFace."""
            # Generate file list like in the notebook
            data_files = [
                f'part-{i:05}-b31ba513-fc6b-4450-9ba4-a1bba183f408-c000.snappy.parquet'
                for i in range(RELAION_NUM_FILES)
            ]
            
            # Load dataset
            ds = load_dataset(
                "laion/relaion2B-en-research",
                cache_dir='./cache',
                data_files=data_files,
                columns=['caption'],  # Only load caption column
                token='***REMOVED***',
            )

            pattern = re.compile(f'(^|[\\s.,-:;]){re.escape(concept.lower().strip())}($|[\\s.,-:;])', flags=re.IGNORECASE) if concept is not None else None
            

            for caption in ds['train']['caption']:
                if caption is None:
                    continue
                if pattern is None:
                    yield caption
                    continue
                if pattern.search(caption) is not None:
                    yield caption

        data = itertools.islice(generator(), max_samples)

        super().__init__(
            data=data,
            tokenizer=tokenizer,
            tokenizer_fn=tokenizer_fn,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
            seed=seed,
        )

class ImageNetDataset(QuestionsDataset):
    """Dataset that loads ImageNet classes and adds ', with {concept}' to the end."""

    def __init__(self,
                 *,
                 concept: str | None = None,
                 max_samples: int | None = None,
                 tokenizer: AutoTokenizer | None = None,
                 tokenizer_fn: tp.Callable = dumb_tokenizer_fn,
                 device: tp.Any = None,
                 instruction: str | None = None,
                 dataset_slice: slice | None = None,
                 seed: int | None = None):
        
        with open('imagenet_classes.txt', 'r') as f:
            data = [f'{line.strip()}, with {concept}' for line in f.readlines()]

        data = itertools.islice(data, max_samples)

        super().__init__(
            data=data,
            tokenizer=tokenizer,
            tokenizer_fn=tokenizer_fn,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
            seed=seed,
        )