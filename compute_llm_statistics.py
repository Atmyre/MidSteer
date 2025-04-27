import argparse
import enum
import time
from steering_vectors import record_activations

from construct_prompts import pickle_stats, read_prompt_file
from controller import VectorControlMode
from llm_utils import ComparisonDataset, init_model_and_tokenizer
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
):
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()

    dataset = ComparisonDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        use_chat=('chat' in model_name),
        device=device,
    )

    pos_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=-2,
    )

    neg_vector_control = CrossAttentionOutputStatsCollector(
        mode=VectorControlMode.ATTN_OUTPUT,
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=-2,
    )

    with record_activations(model, layer_type=layer_type) as records:

        for idx, (p_tokens, n_tokens) in enumerate(tqdm.tqdm(dataset, desc="Processing prompts")):
            if idx in checkpoint_steps:
                pickle_stats(pos_vector_control.means, f'{output_dir}/pos_means_{idx}.pickle')
                pickle_stats(pos_vector_control.covariances, f'{output_dir}/pos_covariances_{idx}.pickle')
                pickle_stats(neg_vector_control.means, f'{output_dir}/neg_means_{idx}.pickle')
                pickle_stats(neg_vector_control.covariances, f'{output_dir}/neg_covariances_{idx}.pickle')

            start = time.time()
            _ = model.forward(p_tokens, use_cache=False)
            print(f'Pos generation took {time.time() - start}')
            for layer_id, record in records.items():
                tensor = record[0].unsqueeze(-2)
                pos_vector_control.forward(tensor, 0, 'LLM', layer_id,)
            records.clear()

            start = time.time()
            _ = model.forward(n_tokens, use_cache=False)
            print(f'Neg generation took {time.time() - start}')
            for layer_id, record in records.items():
                tensor = record[0].unsqueeze(-2)
                neg_vector_control.forward(tensor, 0, 'LLM', layer_id)
            records.clear()

        pickle_stats(pos_vector_control.means, f'{output_dir}/pos_means_{idx}.pickle')
        pickle_stats(pos_vector_control.covariances, f'{output_dir}/pos_covariances_{idx}.pickle')
        pickle_stats(neg_vector_control.means, f'{output_dir}/neg_means_{idx}.pickle')
        pickle_stats(neg_vector_control.covariances, f'{output_dir}/neg_covariances_{idx}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['meta-llama/Llama-2-7b-hf'], required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_steps', type=str, default='100,500,1000,5000')

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        dataset_path=args.dataset_path,
        token_aggregation_mode=args.token_aggregation_mode,
        normalize_vectors=args.normalize_vectors,
        output_dir=args.output_dir,
        checkpoint_steps=set(map(int, args.checkpoint_steps.split(','))),
    )
