import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from core.pickle import unpickle
from core.pickle import unpickle_pack
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

# local imports
from core.controller import CrossAttentionOutputSteering, ModelToSteer, DiffusionVectorControlMode
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks

# parsing arguments
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True)
parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode], default='attn_output', help='Vector control mode')
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--prompt_file', type=str, default=None, help="Path to text file with prompts, one per line.")
parser.add_argument('--seed', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
parser.add_argument('--mmsteer_vectors', type=str, default=None) # path to mmsteer steering vectors file
parser.add_argument('--target_concepts', type=str, default=None)  # path to target concept steering vectors (comma delimited)
parser.add_argument('--source_concepts', type=str, required=True)  # path to source concept steering vectors (comma delimited)
parser.add_argument('--mu_neutral', type=str, default=None)  # path to mu_neutral file
parser.add_argument('--cov', type=str, default=None)  # path to cov file
parser.add_argument('--not_steer', action='store_true')
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--steer_back', action='store_true')
parser.add_argument('--strength', type=float, required=True)
parser.add_argument(
    '--output',
    type=str,
    default='output.png',
    help='Output path to image or directory, in case of multiple images'
)
parser.add_argument('--steer_type', type=str, choices=['casteer', 'mmsteer', 'leace', 'mean_matching', 'interpret'], default=None)
parser.add_argument('--num_images_per_prompt', type=int, default=1)
args = parser.parse_args()

if (args.prompt is not None) == (args.prompt_file is not None):
    raise ValueError("Exactly one of --prompt, --prompt_file should be set")

if args.prompt is not None:
    prompts = [args.prompt]
else:
    with open(args.prompt_file, 'r') as fin:
        prompts = list(map(str.strip, fin.readlines()))

seeds = list(map(int, args.seed.split(",")))


device = get_device()
pipe = init_pipeline_for_image_model(args.model)

if not args.not_steer:
    controller = CrossAttentionOutputSteering(
        model_to_steer=ModelToSteer.UNET,
        mode=args.control_mode,
        mmsteer_vectors=unpickle(args.mmsteer_vectors),
        source_concepts=unpickle_pack(args.source_concepts),
        target_concepts=unpickle_pack(args.target_concepts),
        mu_neutral=unpickle(args.mu_neutral),
        sigma_neutral=unpickle(args.cov),
        steer_type=args.steer_type,
        steer_only_up=args.steer_only_up,
        steer_back=args.steer_back,
        strength=args.strength,
        device=device,
        renormalize_after_steering=True,
        intermediate_clipping=True,
    )
    # Register hooks on the appropriate model component
    model_component = getattr(pipe, 'transformer', None) or pipe.unet
    hook_manager = diffusion_register_vector_controls_with_hooks(
        model_component,
        controller,
        model_type=DiffusionModelType.from_model(args.model),
    )
else:
    controller = None
    hook_manager = None

if args.num_images_per_prompt == 1:
    for prompt in prompts:
        for seed in seeds:
            if len(seeds) == 1 and len(prompts) == 1:
                path = args.output
            else:
                if args.not_steer:
                    file = 'orig.png'
                elif args.steer_back and args.steer_type == 'casteer':
                    file = f'casteer_{args.strength:g}.png'
                else:
                    file = f'{args.steer_type}_{args.strength:g}.png'
                path = f'{args.output}/{prompt}/{seed}/{file}'
            if os.path.exists(path):
                print(f'{path} already exists, skipping!')
                continue
            print(f'Generating for prompt={prompt}, seed={seed}')
            images = run_image_model(args.model, pipe, prompt, seed, device=device, num_images=args.num_images_per_prompt)
            if controller is not None:
                controller.reset()
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            images[0].save(path)
else:
#     if len(seeds) > 1:
#         raise ValueError('num_images_per_prompt > 1 is not supported for multiple seeds')
#     seed = seeds[0]
    
    for seed in seeds:
        for prompt in prompts:

            if args.not_steer:
                file = 'orig.png'
            elif args.steer_back and args.steer_type == 'casteer':
                file = f'casteer_{args.strength:g}.png'
            else:
                file = f'{args.steer_type}_{args.strength:g}.png'
            path = f'{args.output}/{prompt}/{{seed}}/{file}'
            if os.path.exists(path.format(seed=0)):
                print(f'{path} already exists, skipping!')
                continue
            print(f'Generating for prompt={prompt}, seed={seed}')
            images = run_image_model(args.model, pipe, prompt, seed, device=device, num_images=args.num_images_per_prompt)
            if controller is not None:
                controller.reset()
            for i, image in enumerate(images):
                if os.path.dirname(path.format(seed=i)):
                    os.makedirs(os.path.dirname(path.format(seed=i)), exist_ok=True)
                image.save(path.format(seed=seed*args.num_images_per_prompt+i))

# Clean up hooks when done
if hook_manager is not None:
    hook_manager.remove_hooks()