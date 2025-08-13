"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
from CAA.llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from CAA.utils.helpers import get_a_b_probs
from CAA.utils.tokenize import E_INST
from CAA.steering_settings import SteeringSettings
from CAA.behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
    load_cov_neutral,
    load_external_vectors,
    load_mu_neutral,
)
from core.controller import CrossAttentionOutputSteering, ModelToSteer
from core.pickle import unpickle
from core.utils import get_device
import typing as tp

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=100
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def get_CAA_steering_vector(model: LlamaWrapper, settings: SteeringSettings, layer: int):
    name_path = model.model_name_path
    if settings.override_vector_model is not None:
        name_path = settings.override_vector_model
    if settings.override_vector is not None:
        vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
    else:
        vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
    if settings.model_size != "7b":
        vector = vector.half()
    vector = vector.to(model.device)
    return vector


def test_steering(
    model: LlamaWrapper, layers: List[int], multipliers: List[int], settings: SteeringSettings, device: tp.Any,
    mu_neutral: dict,
    cov_neutral: dict,
    renormalize_after_steering: bool,
    overwrite=False,
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        "open_ended": get_open_ended_test_data(settings.behavior),
        "truthful_qa": get_truthful_qa_data(),
        "mmlu": get_mmlu_data(),
    }
    model.set_external_layer_type(settings.external_layer_type)
    model.set_external_layers_to_steer(settings.external_layers_to_steer)
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                if settings.control_type == 'CAA':
                    model.set_add_activations(
                        layer, multiplier * get_CAA_steering_vector(model, settings, layer)
                    )
                elif settings.control_type == 'external':
                    external_controllers = [CrossAttentionOutputSteering(
                        model_to_steer=ModelToSteer.LLAMA,
                        steer_type=settings.external_steer_type,
                        strength=multiplier,
                        steer_back=True,
                        device=device,
                        mu_pos=[load_external_vectors(behavior, settings.external_layer_type, model.model_name_path, 'pos')],
                        mu_neg=[load_external_vectors(behavior, settings.external_layer_type, model.model_name_path, 'neg')],
                        mu_neutral=[mu_neutral],
                        cov=[cov_neutral],
                        renormalize_after_steering=renormalize_after_steering,
                    )]
                    model.set_external_controllers(external_controllers)
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument('--control_type', choices=['CAA', 'external'], default='CAA')
    parser.add_argument('--external_steer_type', choices=['casteer', 'leace'], default=None)
    parser.add_argument('--external_layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], type=str, default=None)
    parser.add_argument('--external_layers_to_steer', type=str, help='Comma separated list of layer indices to steer for external steering (default: steer all layers)', default=None)
    parser.add_argument('--mu_neutral', type=str, default=None)
    parser.add_argument('--cov_neutral', type=str, default=None)
    parser.add_argument('--renormalize_after_steering', action='store_true', help='Renormalize vectors after steering')


    
    args = parser.parse_args()

    mu_neutral = unpickle(args.mu_neutral)
    cov_neutral = unpickle(args.cov_neutral)

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.control_type = args.control_type
    steering_settings.external_steer_type = args.external_steer_type
    steering_settings.external_layer_type = args.external_layer_type
    steering_settings.renormalize_after_steering = args.renormalize_after_steering
    if args.external_layers_to_steer is not None:
        steering_settings.external_layers_to_steer = list(map(int, args.external_layers_to_steer.split(',')))
    else:
        steering_settings.external_layers_to_steer = None

    device = get_device()
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=steering_settings.model_size,
        use_chat=not steering_settings.use_base_model,
        override_model_weights_path=steering_settings.override_model_weights_path,
        control_type=steering_settings.control_type,
        device=device,
    )

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            model=model,
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            device=device,
            overwrite=args.overwrite,
            mu_neutral=mu_neutral,
            cov_neutral=cov_neutral,
            renormalize_after_steering=steering_settings.renormalize_after_steering,
        )
