"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import typing as tp
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from controller import VectorControlMode
from CAA.llama_wrapper import LlamaWrapper
import argparse
from typing import List
from llm_steering import llm_register_vector_control
from utils import get_device
from CAA.utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from CAA.behaviors import (
    get_external_vector_dir,
    get_external_vector_means_path,
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)
from vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, use_chat: bool, device: tp.Any):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.device = device

    def prompt_to_tokens(self, instruction, model_output):
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
        return t.tensor(tokens, device=self.device).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
    external_layer_type: str | None,
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if not os.path.exists(get_external_vector_dir(behavior, external_layer_type)):
        os.makedirs(get_external_vector_dir(behavior, external_layer_type))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_name_path
    )
    dataset = ComparisonDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        use_chat=model.use_chat,
        device=get_device()
    )

    pos_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=TokenAggregationMode.LAST,
        normalize=False,
        last_token_offset=-2,
        compute_covariances=False,
    )

    neg_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=TokenAggregationMode.LAST,
        normalize=False,
        last_token_offset=-2,
        compute_covariances=False,
    )
    with llm_register_vector_control(
            model=model.model,
            control=[pos_vector_control, neg_vector_control],
            layer_type=external_layer_type,
        ), t.no_grad():
        for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
            p_tokens = p_tokens.to(model.device)
            n_tokens = n_tokens.to(model.device)

            model.reset_all()
            pos_vector_control.active = True
            neg_vector_control.active = False
            model.get_logits(p_tokens)
            pos_vector_control.reset()
            for layer in layers:
                p_activations = model.get_last_activations(layer)
                p_activations = p_activations[0, -2, :].detach().cpu()
                pos_activations[layer].append(p_activations)

            model.reset_all()
            pos_vector_control.active = False
            neg_vector_control.active = True
            model.get_logits(n_tokens)
            neg_vector_control.reset()
            for layer in layers:
                n_activations = model.get_last_activations(layer)
                n_activations = n_activations[0, -2, :].detach().cpu()
                neg_activations[layer].append(n_activations)

    pos_vector_control.save_stats(means_path=get_external_vector_means_path(behavior, external_layer_type, model.model_name_path, 'pos'), use_torch_save=True)
    neg_vector_control.save_stats(means_path=get_external_vector_means_path(behavior, external_layer_type, model.model_name_path, 'neg'), use_torch_save=True)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    external_layer_type: str | None,
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model, external_layer_type
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument('--external_layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], default=None)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.external_layer_type,
    )
