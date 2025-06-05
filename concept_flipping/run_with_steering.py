import argparse
import json
import os
from pprint import pprint
from contextlib import contextmanager

import torch
import tqdm

from concept_flipping.dataset import QuestionsDataset, AlpacaDataset, TemplateDataset
from concept_flipping.paths import get_results_path, get_vector_path
from utils import unpickle, unpickle_pack
from transformers import GenerationConfig

from controller import CrossAttentionOutputSteering, ModelToSteer, VectorControlMode
from llm_steering import llm_register_vector_control
from llm_utils import init_model_and_tokenizer
from CAA.utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from utils import get_device


from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


@contextmanager
def dummy_context_manager():
    yield


def main(
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        use_chat: bool,
        layer_type: str,
        layers_to_steer: list[int] | None,
        source_concept: str,
        target_concept: str,
        strength: float,

        max_new_tokens: int,
        mu_neutral: list[dict],
        cov_neutral: list[dict],
        steer_type: str | None,
        alpaca_eval: bool,
        alpaca_num_samples: int | None,
        samples_per_question: int,
        generation_temperature: float,
):
    
    if alpaca_eval:
        dataset = AlpacaDataset(
            data_path=f'concept_flipping/eval/alpaca_instruct/alpaca_data.json',
            tokenizer=tokenizer,
            use_chat=use_chat,
            device=device,
            dataset_slice=slice(0, alpaca_num_samples),
        )
    else:
        dataset = TemplateDataset(
            template_path=f'concept_flipping/eval/concepts/template.json',
            concept=source_concept,
            tokenizer=tokenizer,
            use_chat=use_chat,
            device=device,
        )
    
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, top_k=1)

    if steer_type is not None:
        mu_pos = unpickle(get_vector_path(model_name, layer_type, source_concept))
        mu_neg = unpickle(get_vector_path(model_name, layer_type, target_concept))

        if steer_type == 'mean_matching':
            mu_pos, mu_neg = mu_neg, mu_pos

        control = CrossAttentionOutputSteering(
            mode=VectorControlMode.ATTN_OUTPUT,
            model_to_steer=ModelToSteer.LLAMA,
            steer_type=steer_type,
            steer_back=True,
            device=device,
            mu_pos=[mu_pos],
            mu_neg=[mu_neg],
            mu_neutral=mu_neutral,
            cov=cov_neutral,
            strength=strength,
        )
    else:
        control = None

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=(samples_per_question > 1),
        num_return_sequences=samples_per_question,
        temperature=generation_temperature
    )


    results = []
    for tokens in tqdm.tqdm(dataset, desc=f"Processing prompts for {source_concept} -> {target_concept}"):

        if steer_type is not None:
            context_manager = llm_register_vector_control(
                model=model,
                control=[control],
                layer_type=layer_type,
                layers_to_steer=layers_to_steer,
                min_token_index=tokens.shape[1],
            )
        else:
            context_manager = dummy_context_manager()

        with context_manager, torch.no_grad():
            outputs = model.generate(tokens, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            prompt = tokenizer.decode(token_ids=tokens[0])
            decoded = tokenizer.batch_decode(outputs)

            results.extend([{
                "raw_output": text,
                "prompt": prompt,
                "output": text.split(prompt)[1],
            } for text in decoded])

    output_path = get_results_path(
        model_name=model_name,
        layer_type=layer_type,
        source_concept=source_concept,
        target_concept=target_concept,
        eval_num_samples=alpaca_num_samples,
        steer_type=steer_type,
        strength=strength,
        alpaca_eval=alpaca_eval,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--target_concept', type=str, required=True, help='Name of the positive concept')
    parser.add_argument('--source_concept', type=str, required=True, help='Name of the negative concept')
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj'], nargs='+', required=True)
    parser.add_argument('--layers_to_steer', type=str, help='Comma separated list of layer indices to steer', default=None)
    parser.add_argument('--alpaca_eval', action='store_true', help='Use alpaca eval dataset')
    parser.add_argument('--alpaca_num_samples', type=int, default=None, help='Number of samples to use for alpaca eval')
    parser.add_argument('--samples_per_question', type=int, default=1, help='Number of output samples to generate for each question')
    parser.add_argument('--generation_temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--strength', type=float, default=2.0)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--steer_type', type=str, choices=['casteer', 'leace', 'mean_matching'], default=None)
    parser.add_argument('--mu_neutral', type=str, default=None, help='path to mu_neutral file (for leace and mean_matching)')
    parser.add_argument('--cov_neutral', type=str, default=None, help='path to cov file (for leace and mean_matching)')

    args = parser.parse_args()

    if args.layers_to_steer is not None:
        layers_to_steer = list(map(int, args.layers_to_steer.split(',')))
    else:
        layers_to_steer = None

    model, tokenizer = init_model_and_tokenizer(model_name=args.model_name)
    use_chat = 'chat' in args.model_name
    device = get_device()

    main(
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_chat=use_chat,
        layer_type=args.layer_type,
        layers_to_steer=layers_to_steer,
        source_concept=args.source_concept,
        target_concept=args.target_concept,
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        mu_neutral=unpickle_pack(args.mu_neutral),
        cov_neutral=unpickle_pack(args.cov_neutral),
        steer_type=args.steer_type,
        alpaca_eval=args.alpaca_eval,
        alpaca_num_samples=args.alpaca_num_samples,
        samples_per_question=args.samples_per_question,
        generation_temperature=args.generation_temperature,
    )
