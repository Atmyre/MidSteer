import argparse
import torch
import enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import record_activations

from construct_prompts import pickle_stats, read_prompt_file
from controller import VectorControlMode
from utils import get_device
from vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode


def init_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,   
        cache_dir='../cache',
        torch_dtype=torch.float16,
        device_map='balanced',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='../cache',
        torch_dtype=torch.float16,
        device_map='balanced',
    )
    return model, tokenizer


def main(
        model_name: str,
        layer_type: str,
        prompts: list[str],
        token_aggregation_mode: TokenAggregationMode,
        normalize_vectors: bool,
        output_prefix: str,
        checkpoint_steps: set[int],
):
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()

    vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
    )

    with record_activations(model, layer_type=layer_type) as records:

        for idx, prompt in enumerate(prompts):
            if idx in checkpoint_steps:
                pickle_stats(vector_control.means, f'{output_prefix}_means_{idx}.pickle')
                pickle_stats(vector_control.covariances, f'{output_prefix}_covariances_{idx}.pickle')
            inputs = tokenizer(prompt, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            _ = model.forward(**inputs)

            for layer_id, record in records.items():
                tensor = record[0].unsqueeze(-2)
                vector_control.forward(tensor, 0, 'LLM', layer_id)

            records.clear()
        pickle_stats(vector_control.means, f'{output_prefix}_means_{idx+1}.pickle')
        pickle_stats(vector_control.covariances, f'{output_prefix}_covariances_{idx+1}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['meta-llama/Llama-2-7b-hf'], required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--output_prefix', type=str, required=True)
    parser.add_argument('--checkpoint_steps', type=str, default='100,500,1000,5000')

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        prompts=read_prompt_file(args.prompt_file),
        token_aggregation_mode=args.token_aggregation_mode,
        normalize_vectors=args.normalize_vectors,
        output_prefix=args.output_prefix,
        checkpoint_steps=set(map(int, args.checkpoint_steps.split(','))),
    )
