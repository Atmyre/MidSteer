import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from calculate_mmsteer import unpickle
from utils import get_device, init_pipeline_for_model, run_model

# local imports
from controller import CrossAttentionOutputSteering, VectorControlMode, register_vector_controls

# parsing arguments
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sd14")
parser.add_argument('--control_mode', type=VectorControlMode, choices=[str(x) for x in VectorControlMode], default='attn_output', help='Vector control mode')
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--prompt_file', type=str, default=None, help="Path to text file with prompts, one per line.")
parser.add_argument('--seed', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
parser.add_argument('--casteer_vectors', type=str, default=None) # path to casteer steering vectors file
parser.add_argument('--mmsteer_vectors', type=str, default=None) # path to mmsteer steering vectors file
parser.add_argument('--mu_pos', type=str, default=None)  # path to mu_pos file
parser.add_argument('--mu_neg', type=str, default=None)  # path to mu_neg file
parser.add_argument('--mu_neutral', type=str, default=None)  # path to mu_neutral file
parser.add_argument('--cov', type=str, default=None)  # path to mu_neutral file
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
parser.add_argument('--steer_type', type=str, choices=['casteer', 'mmsteer', 'leace', 'mean_matching'], default=None)
parser.add_argument('--leace_cov', type=str, default=None)
parser.add_argument('--leace_mean', type=str, default=None)
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
pipe = init_pipeline_for_model(args.model)

casteer_vectors = []
if args.casteer_vectors is not None:
    casteer_vectors_names = args.casteer_vectors.split(',')
    for casteer_vector in casteer_vectors_names:
        casteer_vectors.append(unpickle(casteer_vector))

if not args.not_steer:
    controller = CrossAttentionOutputSteering(
        mode=args.control_mode,
        casteer_vectors=casteer_vectors,
        mmsteer_vectors=unpickle(args.mmsteer_vectors),
        leace_cov=unpickle(args.leace_cov),
        leace_mean=unpickle(args.leace_mean),
        mu_pos=unpickle(args.mu_pos),
        mu_neg=unpickle(args.mu_neg),
        mu_neutral=unpickle(args.mu_neutral),
        cov=unpickle(args.cov),
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
                file = f'casteer_{args.beta:g}_sim.png'
            else:
<<<<<<< Updated upstream
                file = f'{args.steer_type}_{args.alpha:g}.png'
=======
                file = f'{args.steer_type}_{args.alpha:g}_2.png'
>>>>>>> Stashed changes
            path = f'{args.output}/{prompt}/{seed}/{file}'
        if os.path.exists(path):
            print(f'{path} already exists, skipping!')
            continue
        print(f'Generating for prompt={prompt}, seed={seed}')
        image = run_model(args.model, pipe, prompt, seed, device=device)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
