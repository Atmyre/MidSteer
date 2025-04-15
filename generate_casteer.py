import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from utils import get_device, init_pipeline_for_model, run_model

# local imports
from controller import CrossAttentionSteering, register_vector_controls

# parsing arguments
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sd14")
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--prompt_file', type=str, default=None, help="Path to text file with prompts, one per line.")
parser.add_argument('--seed', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
parser.add_argument('--casteer_vectors', type=str, default=None) # path to casteer steering vectors file
parser.add_argument('--mmsteer_vectors', type=str, default=None) # path to mmsteer steering vectors file
parser.add_argument('--not_steer', action='store_true')
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--num_denoising_steps', type=int, default=50) # 50 for sd14, sd21, 1 for turbo, 30 for sdxl
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
parser.add_argument('--steer_type', type=str, choices=['casteer', 'mmsteer'], default=None)
args = parser.parse_args()

if (args.prompt is not None) == (args.prompt_file is not None):
    raise ValueError("Exactly one of --prompt, --prompt_file should be set")

if args.prompt is not None:
    prompts = [args.prompt]
else:
    with open(args.prompt_file, 'r') as fin:
        prompts = list(map(str.strip, fin.readlines()))

seeds = list(map(int, args.seed.split(",")))

if args.casteer_vectors is not None:
    with open(args.casteer_vectors, 'rb') as handle:
        casteer_vectors = pickle.load(handle)
else:
    casteer_vectors = None
    
if args.mmsteer_vectors is not None:
    with open(args.mmsteer_vectors, 'rb') as handle:
        mmsteer_vectors = pickle.load(handle)
else:
    mmsteer_vectors = None

device = get_device()
pipe = init_pipeline_for_model(args.model)

if not args.not_steer:
    controller = CrossAttentionSteering(
        casteer_vectors=casteer_vectors,
        mmsteer_vectors=mmsteer_vectors,
        steer_type=args.steer_type,
        mmsteer_threshold=args.mmsteer_thr,
        steer_only_up=args.steer_only_up,
        steer_back=args.steer_back,
        alpha=args.alpha,
        beta=args.beta,
        device=device
    )
    
    register_vector_controls(pipe.unet, controller)

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
        image = run_model(args.model, pipe, prompt, seed, args.num_denoising_steps, device=device)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
