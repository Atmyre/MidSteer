import argparse

import torch

from calculate_mmsteer import unpickle
from transformers import GenerationConfig

from controller import CrossAttentionOutputSteering, VectorControlMode
from llm_steering import llm_register_vector_control
from llm_utils import init_model_and_tokenizer, tokenize_llama_base, tokenize_llama_chat
from utils import get_device


def main(
        model_name: str,
        layer_type: str,
        layers_to_steer: list[int] | None,
        pos_means: dict,
        neg_means: dict,
        prompt: str,
        alpha: float,
        beta: float,
        steer_back: bool,
        max_new_tokens: int,
        mu_pos: dict,
        mu_neg: dict,
        mu_neutral: dict,
        cov: dict,
        steer_type: str,
        leace_cov: dict,
        leace_mean: dict,
):
    
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()

    if pos_means is not None and neg_means is not None:
        num_layers = len(pos_means[0]['LLM'])
        steering_vectors = {0: {'LLM': []}}
        for idx in range(num_layers):
            pos_mean = pos_means[0]['LLM'][idx]
            neg_mean = neg_means[0]['LLM'][idx]

            vec = (pos_mean - neg_mean)
            # vec /= torch.linalg.norm(vec, dim=-1, keepdim=True)
            steering_vectors[0]['LLM'].append(vec)
        steering_vectors = [steering_vectors]
    else:
        steering_vectors=None



    control = CrossAttentionOutputSteering(
        mode=VectorControlMode.ATTN_OUTPUT,
        casteer_vectors=steering_vectors,
        steer_type=steer_type,
        alpha=alpha,
        beta=beta,
        steer_back=steer_back,
        device=device,
        mu_pos=mu_pos,
        mu_neg=mu_neg,
        mu_neutral=mu_neutral,
        cov=cov,
        leace_cov=leace_cov,
        leace_mean=leace_mean,
    )

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

    if 'chat' in model_name:
        inputs = tokenize_llama_chat(tokenizer=tokenizer, user_input=prompt)
    else:
        inputs = tokenize_llama_base(tokenizer=tokenizer, user_input=prompt)
    inputs = torch.tensor(inputs, device=device).unsqueeze(0)

    min_token_index = 0#inputs.shape[1] - 1

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
    parser.add_argument('--pos_means', type=str, default=None)
    parser.add_argument('--neg_means', type=str, default=None)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--steer_back', action='store_true')
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--steer_type', type=str, choices=['casteer', 'mmsteer', 'leace', 'mean_matching'], default=None)
    parser.add_argument('--leace_cov', type=str, default=None)
    parser.add_argument('--leace_mean', type=str, default=None)
    parser.add_argument('--mu_pos', type=str, default=None)  # path to mu_pos file
    parser.add_argument('--mu_neg', type=str, default=None)  # path to mu_neg file
    parser.add_argument('--mu_neutral', type=str, default=None)  # path to mu_neutral file
    parser.add_argument('--cov', type=str, default=None)  # path to mu_neutral file

    args = parser.parse_args()

    if args.layers_to_steer is not None:
        layers_to_steer = list(map(int, args.layers_to_steer.split(',')))
    else:
        layers_to_steer = None

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        layers_to_steer=layers_to_steer,
        pos_means=unpickle(args.pos_means),
        neg_means=unpickle(args.neg_means),
        prompt=args.prompt,
        alpha=args.alpha,
        beta=args.beta,
        steer_back=args.steer_back,
        max_new_tokens=args.max_new_tokens,
        mu_pos=unpickle(args.mu_pos),
        mu_neg=unpickle(args.mu_neg),
        mu_neutral=unpickle(args.mu_neutral),
        cov=unpickle(args.cov),
        steer_type=args.steer_type,
        leace_cov=unpickle(args.leace_cov),
        leace_mean=unpickle(args.leace_mean),
    )
