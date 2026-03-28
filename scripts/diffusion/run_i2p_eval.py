"""
Evaluate steering methods on I2P (Inappropriate Image Prompts) violence subset.
Generates images from I2P violence prompts with different steering configs,
then scores with CLIP for violence/peace classification.

Usage:
    python scripts/diffusion/run_i2p_eval.py \
        --model_name sana \
        --steering_vectors_dir results/sana/violence_peace/steering_vectors \
        --covariances_dir results/sana/violence_peace/covariances \
        --output_dir results/sana/violence_peace/evaluation/i2p \
        --num_prompts 200 --strengths 1.0 2.0 3.0 4.0 5.0
"""
import argparse
import json
import os

import torch
import tqdm
from datasets import load_dataset

from core.controller import CrossAttentionOutputSteering, DiffusionVectorControlMode
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.utils import SUPPORTED_DIFFUSION_MODELS, init_pipeline_for_image_model, run_image_model, get_device


SAVE_OPTIONS = {
    'JPEG': {'subsampling': '4:4:4', 'quality': 95},
}


def generate_images(pipeline, model_name, prompts, output_dir, seed, device, vector_control=None):
    os.makedirs(output_dir, exist_ok=True)
    for i, prompt in enumerate(tqdm.tqdm(prompts, desc=f"Generating {os.path.basename(output_dir)}")):
        out_path = os.path.join(output_dir, f"{i:04d}.jpg")
        if os.path.exists(out_path):
            continue
        images = run_image_model(
            model_type=model_name,
            pipe=pipeline,
            prompt=prompt,
            seed=seed,
            device=device,
            num_images=1,
        )
        if vector_control is not None:
            vector_control.reset()
        images[0].save(out_path, format='JPEG', **SAVE_OPTIONS['JPEG'])


def main(args):
    device = get_device()

    # Load I2P violence prompts
    print("Loading I2P dataset...")
    ds = load_dataset("AIML-TUDA/i2p", split="train")
    violence_prompts = [row["prompt"] for row in ds if "violence" in row["categories"].lower()]
    print(f"Found {len(violence_prompts)} violence prompts in I2P")

    if args.num_prompts and args.num_prompts < len(violence_prompts):
        violence_prompts = violence_prompts[:args.num_prompts]
    print(f"Using {len(violence_prompts)} prompts")

    # Save prompts for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prompts.json"), "w") as f:
        json.dump(violence_prompts, f, indent=2)

    # Load model
    print(f"Loading {args.model_name}...")
    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)

    # Load steering components
    source_concept = unpickle(os.path.join(args.steering_vectors_dir, "violence.pt"))
    target_concept = unpickle(os.path.join(args.steering_vectors_dir, "peace.pt"))
    mu_neutral = unpickle(os.path.join(args.covariances_dir, "means.pt"))
    cov_neutral = unpickle(os.path.join(args.covariances_dir, "covariances.pt"))

    # Generate baseline
    print("=== Baseline ===")
    generate_images(pipeline, args.model_name, violence_prompts,
                   os.path.join(args.output_dir, "orig"), args.seed, device)

    # Generate steered versions
    for method in ["casteer", "leace", "midsteer"]:
        for strength in args.strengths:
            label = f"{method}-{strength}"
            print(f"=== {label} ===")

            vector_control = CrossAttentionOutputSteering(
                model_to_steer="unet",
                mode=args.control_mode,
                steer_type=method,
                target_concepts=[target_concept],
                source_concepts=[source_concept],
                mu_neutral=mu_neutral,
                sigma_neutral=cov_neutral,
                steer_only_up=False,
                steer_back=True,
                strength=strength,
                device=device,
                use_first_diffusion_step=True,
            )

            model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
            diffusion_register_vector_controls_with_hooks(
                model_component,
                vector_control,
                model_type=DiffusionModelType.from_model(args.model_name),
            )

            generate_images(pipeline, args.model_name, violence_prompts,
                           os.path.join(args.output_dir, label), args.seed, device,
                           vector_control=vector_control)

            # Remove hooks for next iteration
            for handle in getattr(vector_control, '_hook_handles', []):
                handle.remove()

    print(f"\n=== I2P evaluation COMPLETE ===")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=SUPPORTED_DIFFUSION_MODELS)
    parser.add_argument("--steering_vectors_dir", type=str, required=True)
    parser.add_argument("--covariances_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--control_mode", type=str, default="attn_output")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    args = parser.parse_args()
    main(args)
