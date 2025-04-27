import argparse

import torch

from calculate_mmsteer import unpickle
from transformers import GenerationConfig

from controller import CrossAttentionOutputSteering, VectorControlMode
from llm_steering import llm_patch_activations
from llm_utils import init_model_and_tokenizer
from utils import get_device


def main(
        model_name: str,
        layer_type: str,
        pos_means: dict,
        neg_means: dict,
        prompt: str,
        alpha=0.0,
        beta=2.0,
        steer_back=False
):
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)
    device = get_device()


    num_layers = len(pos_means[0]['LLM'])
    steering_vectors = {0: {'LLM': []}}
    for idx in range(num_layers):
        pos_mean = pos_means[0]['LLM'][idx]
        neg_mean = neg_means[0]['LLM'][idx]

        vec = (pos_mean - neg_mean)
        vec /= torch.linalg.norm(vec, dim=-1, keepdim=True)
        steering_vectors[0]['LLM'].append(vec)



    control = CrossAttentionOutputSteering(
        mode=VectorControlMode.ATTN_OUTPUT,
        casteer_vectors=[steering_vectors],
        steer_type='casteer',
        alpha=alpha,
        beta=beta,
        steer_back=steer_back,
        device=device,
        num_layers=num_layers,
    )

    generation_config = GenerationConfig(max_new_tokens=40)


    handle = llm_patch_activations(
        model=model,
        control=control,
        layer_type=layer_type,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    outputs = model.generate(**inputs, generation_config=generation_config)
    print(tokenizer.decode(token_ids=outputs[0]))

    handle.remove()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['meta-llama/Llama-2-7b-hf'], required=True)
    parser.add_argument('--layer_type', choices=['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'], required=True)
    parser.add_argument('--pos_means', type=str, required=True)
    parser.add_argument('--neg_means', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--steer_back', action='store_true')
    parser.add_argument('--beta', type=float, default=2)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        layer_type=args.layer_type,
        pos_means=unpickle(args.pos_means),
        neg_means=unpickle(args.neg_means),
        prompt=args.prompt,
        alpha=args.alpha,
        beta=args.beta,
        steer_back=args.steer_back
    )
