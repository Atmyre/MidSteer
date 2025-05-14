import argparse
import torch
from pathlib import Path

from concept_flipping.paths import get_cov_path
from controller import VectorControlMode
from llm_steering import llm_register_vector_control
from llm_utils import init_model_and_tokenizer
from utils import get_device
from vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
from concept_flipping.dataset import AlpacaDataset
import tqdm


def main(
        model_name: str,
        layer_type: str,
        token_aggregation_mode: TokenAggregationMode,
        normalize_vectors: bool,
        last_token_offset: int,
        num_samples: int | None = None,
):
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()

    dataset = AlpacaDataset(
        data_path=f'concept_flipping/eval/alpaca_instruct/alpaca_data.json',
        tokenizer=tokenizer,
        device=device,
        dataset_slice=slice(-num_samples, None),  # Estimate covariance on last num_samples examples to avoid bias
    )

    vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=last_token_offset,
        compute_covariances=True,
    )

    with llm_register_vector_control(
        model=model,
        control=[vector_control],
        layer_type=layer_type,
    ), torch.no_grad():
        for tokens in tqdm.tqdm(dataset, desc="Processing prompts"):
            vector_control.active = True
            _ = model.forward(tokens, use_cache=False)
            vector_control.reset()

        vector_control.save_stats(
            means_path=get_cov_path(model_name, layer_type, 'means', num_samples),
            covariances_path=get_cov_path(model_name, layer_type, 'covariances', num_samples),
            use_torch_save=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--last_token_offset', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()

    # Create cov directory if it doesn't exist
    Path('concept_flipping/cov').mkdir(parents=True, exist_ok=True)

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        token_aggregation_mode=args.token_aggregation_mode,
        normalize_vectors=args.normalize_vectors,
        last_token_offset=args.last_token_offset,
        num_samples=args.num_samples,
    )