import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from utils import get_device, init_pipeline_for_model

# local imports
from controller import VectorStore, register_vector_control

# parsing arguments
import argparse


def run_model(model_type, pipe, prompt, seed, num_denoising_steps, device):
    if model_type in ['sd14', 'sd21', 'sdxl']:
        image = pipe(prompt=prompt, 
                     num_inference_steps=num_denoising_steps, 
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
      
    elif model_type in ['sd21-turbo', 'sdxl-turbo']:
        image = pipe(prompt=prompt, 
                     num_inference_steps=num_denoising_steps,
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
            
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sd14")
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--prompt_file', type=str, default=None, help="Path to text file with prompts, one per line.")
parser.add_argument('--seed', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
parser.add_argument('--steering_vectors', type=str, default=None) # path to steering vectors file
parser.add_argument('--not_steer', action='store_true')
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--num_denoising_steps', type=int, default=50) # 50 for sd14, sd21, 1 for turbo, 30 for sdxl
parser.add_argument('--steer_back', action='store_true')
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

if args.steering_vectors is not None:
    with open(args.steering_vectors, 'rb') as handle:
        steering_vectors = pickle.load(handle)
else:
    steering_vectors = None

device = get_device()
controller = VectorStore(
    steering_vectors=steering_vectors,
    steer=not args.not_steer,
    steer_type=args.steer_type,
    steer_only_up=args.steer_only_up,
    steer_back=args.steer_back,
    alpha=args.alpha,
    beta=args.beta,
    device=device
)

pipe = init_pipeline_for_model(args.model)
register_vector_control(pipe.unet, controller)

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
