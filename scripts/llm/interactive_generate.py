import argparse

import torch

from core.pickle import unpickle
from core.pickle import unpickle_pack
from transformers import GenerationConfig

from core.controller import CrossAttentionOutputSteering, ModelToSteer, SteeringVectors
from core.llm_steering import llm_register_vector_control
from core.utils import init_llm_model_and_tokenizer
from CAA.utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from core.utils import get_device


def main(
        model_name: str,
        layer_type: str,
        layers_to_steer: list[int] | None,
        prompt: str,
        strength: float,
        steer_back: bool,
        max_new_tokens: int,
        source_concepts: list[SteeringVectors],
        target_concepts: list[SteeringVectors | None],
        mu_neutral: SteeringVectors,
        cov: SteeringVectors,
        steer_type: str,
):
    
    model, tokenizer = init_llm_model_and_tokenizer(model_name=model_name)
    device = get_device()


    control = CrossAttentionOutputSteering(
        model_to_steer=ModelToSteer.LLAMA,
        steer_type=steer_type,
        strength=strength,
        steer_back=steer_back,
        device=device,
        source_concepts=source_concepts,
        target_concepts=target_concepts,
        mu_neutral=mu_neutral,
        sigma_neutral=cov,
        renormalize_after_steering=True,
        intermediate_clipping=True,
    )

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

    if 'chat' in model_name:
        inputs = tokenize_llama_chat(tokenizer=tokenizer, user_input=prompt)
    else:
        inputs = tokenize_llama_base(tokenizer=tokenizer, user_input=prompt, model_output='')
    inputs = torch.tensor(inputs, device=device).unsqueeze(0)

    min_token_index = inputs.shape[1] - 1

    with llm_register_vector_control(
        model=model,
        control=[control],
        layer_type=layer_type,
        layers_to_steer=layers_to_steer,
        min_token_index=min_token_index,
    ), torch.no_grad():
        outputs = model.generate(inputs, generation_config=generation_config)
        print(tokenizer.decode(token_ids=outputs[0]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--layers_to_steer', type=str, help='Comma separated list of layer indices to steer', default=None)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--steer_back', action='store_true')
    parser.add_argument('--strength', required=True, type=float)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--steer_type', type=str, choices=['casteer', 'mmsteer', 'leace', 'mean_matching'], default=None)
    parser.add_argument('--source_concepts', type=str, required=True)  # path to mu_pos file
    parser.add_argument('--target_concepts', type=str, default=None)  # path to mu_neg file
    parser.add_argument('--mu_neutral', type=str, default=None)  # path to mu_neutral file
    parser.add_argument('--cov', type=str, default=None)  # path to sigma_neural file

    args = parser.parse_args()

    if args.layers_to_steer is not None:
        layers_to_steer = list(map(int, args.layers_to_steer.split(',')))
    else:
        layers_to_steer = None

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        layers_to_steer=layers_to_steer,
        prompt=args.prompt,
        strength=args.strength,
        steer_back=args.steer_back,
        max_new_tokens=args.max_new_tokens,
        source_concepts=unpickle_pack(args.source_concepts),
        target_concepts=unpickle_pack(args.target_concepts),
        mu_neutral=unpickle(args.mu_neutral),
        cov=unpickle(args.cov),
        steer_type=args.steer_type,
    )
