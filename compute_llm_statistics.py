import argparse
import enum
import time
from steering_vectors import record_activations
import torch

from construct_prompts import pickle_stats, read_prompt_file
from controller import VectorControlMode
from llm_steering import llm_register_vector_control
from llm_utils import ComparisonDataset, AlpacaDataset, init_model_and_tokenizer
from utils import get_device
from vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
import tqdm


def main(
        model_name: str,
        layer_type: str,
        dataset_path: str,
        token_aggregation_mode: TokenAggregationMode,
        normalize_vectors: bool,
        output_dir: str,
        checkpoint_steps: set[int],
        last_token_offset: int,
        compute_covariances: bool,
):
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()

    dataset = AlpacaDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        use_chat=('chat' in model_name),
        device=device,
    )

    pos_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=last_token_offset,
        compute_covariances=compute_covariances,
    )

    neg_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=last_token_offset,
        compute_covariances=compute_covariances,
    )

    with llm_register_vector_control(
        model=model,
        control=[pos_vector_control, neg_vector_control],
        layer_type=layer_type,
    ), torch.no_grad():

        for idx, (p_tokens, n_tokens) in enumerate(tqdm.tqdm(dataset, desc="Processing prompts")):
            if idx in checkpoint_steps:
                pos_vector_control.pickle_stats(means_path=f'{output_dir}/pos_means_{idx}.pickle',
                                                covariances_path=f'{output_dir}/pos_covariances_{idx}.pickle')
                neg_vector_control.pickle_stats(means_path=f'{output_dir}/neg_means_{idx}.pickle',
                                                covariances_path=f'{output_dir}/neg_covariances_{idx}.pickle')

            pos_vector_control.active = True
            neg_vector_control.active = False
            _ = model.forward(p_tokens, use_cache=False)
            pos_vector_control.reset()

            pos_vector_control.active = False
            neg_vector_control.active = True
            _ = model.forward(n_tokens, use_cache=False)
            neg_vector_control.reset()

        pos_vector_control.pickle_stats(means_path=f'{output_dir}/pos_means_{idx+1}.pickle',
                                        covariances_path=f'{output_dir}/pos_covariances_{idx+1}.pickle')
        neg_vector_control.pickle_stats(means_path=f'{output_dir}/neg_means_{idx+1}.pickle',
                                        covariances_path=f'{output_dir}/neg_covariances_{idx+1}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_steps', type=str, default='100,500,1000,5000,10000,25000')
    parser.add_argument('--last_token_offset', type=int, default=-1)
    parser.add_argument('--compute_covariances', action='store_true')

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        dataset_path=args.dataset_path,
        token_aggregation_mode=args.token_aggregation_mode,
        normalize_vectors=args.normalize_vectors,
        output_dir=args.output_dir,
        checkpoint_steps=set(map(int, args.checkpoint_steps.split(','))),
        last_token_offset=args.last_token_offset,
        compute_covariances=args.compute_covariances,
    )
