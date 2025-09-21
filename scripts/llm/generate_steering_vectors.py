import argparse
import os
import torch

from core.dataset import ALPACA_DEFAULT_INSTRUCTION, QuestionsDataset, resolve_tokenizer_for_model

from core.llm_steering import llm_register_vector_control
from core.utils import init_llm_model_and_tokenizer
from core.utils import get_device
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
import tqdm


from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

def compute_vectors(
        model: AutoModelForCausalLM,
        model_name: str,
        tokenizer: AutoTokenizer,
        device: torch.device,
        layer_type: str,
        topic: str,
        token_aggregation_mode: TokenAggregationMode,
        last_token_offset: int,
        normalize_vectors: bool,
        max_new_tokens: int,
        num_samples: int,
        output_dir: str,
        use_alpaca_system_prompt: bool,
):
    output_path = os.path.join(output_dir, f"{topic}.pt")
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping generation.")
        return
    os.makedirs(output_dir, exist_ok=True)

    dataset = QuestionsDataset(
        data_path=f'exp/datasets/train/{topic}_questions_batch_*.json',
        tokenizer=tokenizer,
        tokenizer_fn=resolve_tokenizer_for_model(model_name),
        device=device,
        dataset_slice=slice(0, num_samples),
        instruction=ALPACA_DEFAULT_INSTRUCTION if use_alpaca_system_prompt else None,
    )

    vector_control = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=token_aggregation_mode,
        normalize=normalize_vectors,
        last_token_offset=last_token_offset,
        compute_covariances=False,
    )

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

    for tokens in tqdm.tqdm(dataset, desc=f"Processing prompts for {topic}"):
        with llm_register_vector_control(
            model=model,
            control=[vector_control],
            layer_type=layer_type,
            min_token_index=tokens.shape[1] - 1,
        ), torch.no_grad():
            _ = model.generate(tokens, generation_config=generation_config)
            vector_control.reset()
    vector_control.save_stats(
        means_path=output_path,
        use_torch_save=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj'], nargs='+', required=True)
    parser.add_argument('--topics', type=str, nargs='+', required=True, help='List of topics to generate steering vectors for')
    parser.add_argument('--token_aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode], required=True)
    parser.add_argument('--normalize_vectors', action='store_true')
    parser.add_argument('--last_token_offset', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_alpaca_system_prompt', action='store_true', help='Use alpaca instruction for eval set')
    args = parser.parse_args()


    model, tokenizer = init_llm_model_and_tokenizer(model_name=args.model_name)
    device = get_device()

    for topic in args.topics:
        compute_vectors(
            model=model,
            model_name=args.model_name,
            tokenizer=tokenizer,
            device=device,
            layer_type=args.layer_type,
            topic=topic,
            token_aggregation_mode=args.token_aggregation_mode,
            normalize_vectors=args.normalize_vectors,
            last_token_offset=args.last_token_offset,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            use_alpaca_system_prompt=args.use_alpaca_system_prompt,
        )
