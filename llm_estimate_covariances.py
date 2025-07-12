import argparse
import os
import torch
from pathlib import Path


from core.llm_steering import llm_register_vector_control
from utils import init_llm_model_and_tokenizer
from utils import get_device
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
from dataset import ALPACA_DEFAULT_INSTRUCTION, AlpacaDataset
from transformers import GenerationConfig
import tqdm


def main(
        model_name: str,
        layer_type: list[str],
        token_aggregation_mode: TokenAggregationMode,
        normalize_vectors: bool,
        last_token_offset: int,
        output_dir: str,
        num_samples: int | None = None,
        use_chat: bool = False,
        max_new_tokens: int = 1,
        do_not_use_alpaca_system_prompt: bool = False,
):
    if os.path.exists(os.path.join(output_dir, "means.pt")) and os.path.exists(os.path.join(output_dir, "covariances.pt")):
        print(f"File {output_dir}/means.pt and {output_dir}/covariances.pt already exist. Skipping estimation.")
        return
    model, tokenizer = init_llm_model_and_tokenizer(model_name=model_name)
    device = get_device()

    dataset = AlpacaDataset(
        data_path=f'llm_exp/datasets/eval/alpaca_instruct/alpaca_data.json',
        tokenizer=tokenizer,
        use_chat=use_chat,
        device=device,
        dataset_slice=slice(-num_samples, None),  # Estimate covariance on last num_samples examples to avoid bias
        include_model_output=True,
        instruction=None if do_not_use_alpaca_system_prompt else ALPACA_DEFAULT_INSTRUCTION,
    )

    vector_control = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=last_token_offset,
        compute_covariances=True,
    )

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

    for tokens in tqdm.tqdm(dataset, desc=f"Processing prompts"):
        with llm_register_vector_control(
            model=model,
            control=[vector_control],
            layer_type=layer_type,
            min_token_index=tokens.shape[1] - 1,
        ), torch.no_grad():
            _ = model.generate(tokens, generation_config=generation_config)
            vector_control.reset()
    os.makedirs(output_dir, exist_ok=True)
    vector_control.save_stats(
        means_path=os.path.join(output_dir, "means.pt"),
        covariances_path=os.path.join(output_dir, "covariances.pt"),
        use_torch_save=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj'], nargs='+', required=True)
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--last_token_offset', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--do_not_use_alpaca_system_prompt', action='store_true', help='Use alpaca instruction for eval set')

    args = parser.parse_args()

    use_chat = 'chat' in args.model_name

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        token_aggregation_mode=args.token_aggregation_mode,
        normalize_vectors=args.normalize_vectors,
        last_token_offset=args.last_token_offset,
        num_samples=args.num_samples,
        use_chat=use_chat,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        do_not_use_alpaca_system_prompt=args.do_not_use_alpaca_system_prompt,
    )
