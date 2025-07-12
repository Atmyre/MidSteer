import argparse
import json
import os
from pprint import pprint
from contextlib import contextmanager

import torch
import tqdm

from dataset import ALPACA_DEFAULT_INSTRUCTION, MMLU_SYSTEM_PROMPT, QuestionsDataset, AlpacaDataset, TemplateDataset, MMLUDataset
from core.pickle import unpickle
from core.pickle import unpickle_pack
from transformers import GenerationConfig

from core.controller import CrossAttentionOutputSteering, ModelToSteer
from core.llm_steering import llm_register_vector_control
from utils import init_llm_model_and_tokenizer
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
        layer_type: list[str],
        layers_to_steer: list[int] | None,
        source_concept: str,
        source_concept_path: str,
        target_concept_path: str,
        strength: float,
        output_dir: str,

        max_new_tokens: int,
        mu_neutral: list[dict],
        cov_neutral: list[dict],
        steer_type: str | None,
        dataset_type: str,
        num_samples: int | None,
        samples_per_question: int,
        generation_temperature: float,
        identity_cov: bool,
        system_prompt: str | None,
        zero_mu_neutral: bool,
        mm_normalize_centers: bool,
        intermediate_clipping: bool,
        renormalize_after_steering: bool,
):
    output_path = os.path.join(output_dir, f'{steer_type}_{strength:.1f}.json')
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping generation.")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset_slice based on num_samples
    dataset_slice = slice(0, num_samples) if num_samples is not None else None

    # Set default system prompt based on dataset type if not specified
    if system_prompt is None:
        if dataset_type == 'alpaca':
            system_prompt = 'alpaca'
        elif dataset_type == 'mmlu':
            system_prompt = 'mmlu'
        elif dataset_type == 'template':
            system_prompt = None  # No system prompt for template dataset

    # Convert system_prompt string to actual prompt text
    instruction = None
    if system_prompt == 'alpaca':
        instruction = ALPACA_DEFAULT_INSTRUCTION
    elif system_prompt == 'mmlu':
        instruction = MMLU_SYSTEM_PROMPT
    elif system_prompt is not None:
        instruction = system_prompt  # Use custom prompt if provided

    if dataset_type == 'alpaca':
        dataset = AlpacaDataset(
            data_path=f'llm_exp/datasets/eval/alpaca_instruct/alpaca_data.json',
            tokenizer=tokenizer,
            use_chat=use_chat,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
        )
    elif dataset_type == 'template':
        dataset = TemplateDataset(
            template_path=f'llm_exp/datasets/eval/concepts/template.json',
            concept=source_concept,
            tokenizer=tokenizer,
            use_chat=use_chat,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
        )
    elif dataset_type == 'mmlu':
        dataset = MMLUDataset(
            data_path=f'llm_exp/datasets/eval/mmlu/mmlu_full.json',
            tokenizer=tokenizer,
            use_chat=use_chat,
            device=device,
            instruction=instruction,
            dataset_slice=dataset_slice,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if steer_type is not None:
        # TODO: proper
        mu_pos = unpickle(source_concept_path)
        mu_neg = unpickle(target_concept_path)

        if steer_type == 'mean_matching':
            mu_pos, mu_neg = mu_neg, mu_pos

        control = CrossAttentionOutputSteering(
            model_to_steer=ModelToSteer.LLAMA,
            steer_type=steer_type,
            steer_back=True,
            device=device,
            mu_pos=[mu_pos],
            mu_neg=[mu_neg],
            mu_neutral=mu_neutral,
            cov=cov_neutral,
            strength=strength,
            identity_cov=identity_cov,
            zero_mu_neutral=zero_mu_neutral,
            mm_normalize_centers=mm_normalize_centers,
            intermediate_clipping=intermediate_clipping,
            renormalize_after_steering=renormalize_after_steering,
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
    for tokens in tqdm.tqdm(dataset, desc=f"Processing prompts for source concept {source_concept}"):

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
                "prompt": prompt,
                "output": text.split(prompt)[1],
            } for text in decoded])

    with open(output_path, 'w') as f:
        json.dump(results, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--target_concept_path', type=str, required=False, help='Path to the target concept pt file')
    parser.add_argument('--source_concept', type=str, required=True, help='Name of the source concept')
    parser.add_argument('--source_concept_path', type=str, required=False, help='Path to the source concept pt file')
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm', 'q_proj', 'k_proj', 'v_proj', 'o_proj'], nargs='+', required=True)
    parser.add_argument('--layers_to_steer', type=str, help='Comma separated list of layer indices to steer', default=None)
    parser.add_argument('--dataset_type', type=str, choices=['alpaca', 'template', 'mmlu'], required=True)
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use from the dataset (default: use all samples)')
    parser.add_argument('--samples_per_question', type=int, default=1, help='Number of output samples to generate for each question')
    parser.add_argument('--generation_temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--strength', type=float, default=2.0)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--steer_type', type=str, choices=['casteer', 'leace', 'mean_matching'], default=None)
    parser.add_argument('--mu_neutral', type=str, default=None, help='path to mu_neutral file (for leace and mean_matching)')
    parser.add_argument('--cov_neutral', type=str, default=None, help='path to cov file (for leace and mean_matching)')
    parser.add_argument('--identity_cov', action='store_true', help='Use identity covariance matrix for leace and mean_matching')
    parser.add_argument('--zero_mu_neutral', action='store_true', help='Use zero mu_neutral for leace and mean_matching')
    parser.add_argument('--mm_normalize_centers', action='store_true', help='Normalize mean_matching centers')
    parser.add_argument('--intermediate_clipping', action='store_true', help='Apply intermediate clipping like CASteer for leace and mean_matching')
    parser.add_argument('--renormalize_after_steering', action='store_true', help='Renormalize vectors after steering for leace and mean_matching')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, choices=['alpaca', 'mmlu'], default=None, help='System prompt type (default: auto-detect based on dataset type)')

    args = parser.parse_args()

    if args.layers_to_steer is not None:
        layers_to_steer = list(map(int, args.layers_to_steer.split(',')))
    else:
        layers_to_steer = None

    model, tokenizer = init_llm_model_and_tokenizer(model_name=args.model_name)
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
        source_concept_path=args.source_concept_path,
        target_concept_path=args.target_concept_path,
        strength=args.strength,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        mu_neutral=unpickle_pack(args.mu_neutral),
        cov_neutral=unpickle_pack(args.cov_neutral),
        identity_cov=args.identity_cov,
        steer_type=args.steer_type,
        dataset_type=args.dataset_type,
        num_samples=args.num_samples,
        samples_per_question=args.samples_per_question,
        generation_temperature=args.generation_temperature,
        system_prompt=args.system_prompt,
        zero_mu_neutral=args.zero_mu_neutral,
        mm_normalize_centers=args.mm_normalize_centers,
        intermediate_clipping=args.intermediate_clipping,
        renormalize_after_steering=args.renormalize_after_steering,
    )
