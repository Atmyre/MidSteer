import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from core.pickle import unpickle
from core.pickle import unpickle_pack
from utils import get_device, init_pipeline_for_image_model, run_image_model

# local imports
from core.controller_hooks import CrossAttentionOutputSteeringHook, ModelToSteer, DiffusionVectorControlMode, register_vector_controls_with_hooks

# parsing arguments
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo', 'flux', 'flux-schnell'], default="sd14")
parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode], default='attn_output', help='Vector control mode')
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--prompt_file', type=str, default=None, help="Path to text file with prompts, one per line.")
parser.add_argument('--seed', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
parser.add_argument('--mmsteer_vectors', type=str, default=None) # path to mmsteer steering vectors file
parser.add_argument('--mu_pos', type=str, default=None)  # path to mu_pos file
parser.add_argument('--mu_neg', type=str, default=None)  # path to mu_neg file
parser.add_argument('--mu_neutral', type=str, default=None)  # path to mu_neutral file
parser.add_argument('--cov', type=str, default=None)  # path to cov file
parser.add_argument('--not_steer', action='store_true')
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--steer_back', action='store_true')
parser.add_argument('--mmsteer_thr', type=float, default=0)
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--beta', type=float, default=2)
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
    controller = CrossAttentionOutputSteeringHook(
        model_to_steer=ModelToSteer.UNET,
        mode=args.control_mode,
        mmsteer_vectors=unpickle(args.mmsteer_vectors),
        mu_pos=unpickle_pack(args.mu_pos),
        mu_neg=unpickle_pack(args.mu_neg),
        mu_neutral=unpickle_pack(args.mu_neutral),
        cov=unpickle_pack(args.cov),
        steer_type=args.steer_type,
        mmsteer_threshold=args.mmsteer_thr,
        steer_only_up=args.steer_only_up,
        steer_back=args.steer_back,
        alpha=args.alpha,
        beta=args.beta,
        device=device
    )
    try:
        hook_manager = register_vector_controls_with_hooks(pipe.unet, controller)
    except:
        hook_manager = register_vector_controls_with_hooks(pipe.transformer, controller)
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
                    file = f'casteer_{args.beta:g}.png'
                else:
                    file = f'{args.steer_type}_{args.alpha:g}.png'
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
                file = f'casteer_{args.beta:g}.png'
            else:
                file = f'{args.steer_type}_{args.alpha:g}.png'
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