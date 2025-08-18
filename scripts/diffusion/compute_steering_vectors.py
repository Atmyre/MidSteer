import os
import numpy as np
import tqdm
import pickle
from PIL import Image
from collections import defaultdict
import time
import typing as tp
from scipy.linalg import fractional_matrix_power

import torch

# local imports
from core.prompts import get_prompts_concrete, get_prompts_style, get_prompts_human_related, read_prompt_file
# from core.controller import DiffusionVectorControlMode, register_vector_controls
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.controller import DiffusionVectorControlMode
from core.math import fractional_matrix_power_cov_torch
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

# parsing arguments
import argparse
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode



def gather_stats_for_prompts(
        pipe,
        prompts: list[str],
        model_type: str,
        device: torch.device,
        patch_average: bool,
        output_dir: str,
        normalize_vectors: bool,
        checkpoint_steps: list[int],
        control_mode: DiffusionVectorControlMode,
) -> tuple[np.ndarray, np.ndarray]:
    stats_handler = CrossAttentionOutputStatsCollector(
        mode=control_mode,
        token_aggregation_mode=TokenAggregationMode.AVERAGE if patch_average else TokenAggregationMode.ALL,
        normalize=normalize_vectors,
        compute_covariances=True,
    )
    # Register hooks on the appropriate model component
    model_component = getattr(pipe, 'transformer', None) or pipe.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        stats_handler,
        model_type=DiffusionModelType.from_model(model_type),
    )

    print("Gathering statistics for concept prompts...")
    for idx, prompt in tqdm.tqdm(enumerate(prompts), total=len(prompts)):

        if idx in checkpoint_steps:
            stats_handler.save_stats(means_path=f'{output_dir}/means_{idx}.pickle',
                                       covariances_path=f'{output_dir}/covariances_{idx}.pickle')
    
        image = run_image_model(
            model_type=model_type,
            pipe=pipe,
            prompt=prompt,
            seed=0,
            device=device,
        )[0]
        stats_handler.reset()

    stats_handler.save_stats(means_path=f'{output_dir}/means_{idx+1}.pickle',
                               covariances_path=f'{output_dir}/covariances_{idx+1}.pickle')
    


def gather_stats_for_prompt_pairs(
        pipe,
        args: argparse.Namespace,
        checkpoint_steps: list[int],
        device: tp.Any,
        prompts_pos: list[str],
        prompts_neg: list[str],
):
    
    pos_stats_handler = CrossAttentionOutputStatsCollector(
        mode=args.control_mode,
        token_aggregation_mode=TokenAggregationMode.AVERAGE if args.patch_average else TokenAggregationMode.ALL,
        normalize=args.normalize_vectors,
        compute_covariances=False,
    )
    neg_stats_handler = CrossAttentionOutputStatsCollector(
        mode=args.control_mode,
        token_aggregation_mode=TokenAggregationMode.AVERAGE if args.patch_average else TokenAggregationMode.ALL,
        normalize=args.normalize_vectors,
        compute_covariances=False,
    )
    
    # Register hooks on the appropriate model component
    model_component = getattr(pipe, 'transformer', None) or pipe.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        pos_stats_handler,
        neg_stats_handler,
        model_type=DiffusionModelType.from_model(args.model),
    )

    print("Gathering statistics for concept prompts...")
    for idx, (pos_prompt, neg_prompt) in tqdm.tqdm(
            enumerate(zip(prompts_pos, prompts_neg)),
            total=min(len(prompts_pos), len(prompts_neg))
    ):
        if idx in checkpoint_steps:
            write_checkpoint(
                output_dir=args.output_dir,
                step=idx,
                pos_handler=pos_stats_handler,
                neg_handler=neg_stats_handler,
            )

        pos_stats_handler.active = True
        neg_stats_handler.active = False
        image = run_image_model(
            model_type=args.model,
            pipe=pipe,
            prompt=pos_prompt,
            seed=0,
            device=device,
        )[0]
        pos_stats_handler.reset()

        pos_stats_handler.active = False
        neg_stats_handler.active = True
        image = run_image_model(
            model_type=args.model,
            pipe=pipe,
            prompt=neg_prompt,
            seed=0,
            device=device,
        )[0]
        neg_stats_handler.reset()

    write_checkpoint(
        output_dir=args.output_dir,
        step=idx+1,
        pos_handler=pos_stats_handler,
        neg_handler=neg_stats_handler,
    )


def write_checkpoint(
        output_dir: str,
        step: int,
        pos_handler: CrossAttentionOutputStatsCollector,
        neg_handler: CrossAttentionOutputStatsCollector,
):
    pos_handler.save_stats(means_path=f'{output_dir}/pos_means_{step}.pickle',
                             covariances_path=f'{output_dir}/pos_covariances_{step}.pickle')
    neg_handler.save_stats(means_path=f'{output_dir}/neg_means_{step}.pickle',
                             covariances_path=f'{output_dir}/neg_covariances_{step}.pickle')

    casteer_vectors = calculate_casteer(pos_means=pos_handler.means, neg_means=neg_handler.means)
    with open(f'{output_dir}/casteer_{step}.pickle', 'wb') as fout:
        pickle.dump(casteer_vectors, fout)

#     mmsteer_transforms_forward = calculate_mmster(
#         pos_means=pos_handler.means,
#         pos_covariances=pos_handler.covariances,
#         neg_means=neg_handler.means,
#         neg_covariances=neg_handler.covariances,
#     )
#     with open(f'{output_dir}/mmsteer_forward_{step}.pickle', 'wb') as fout:
#         pickle.dump(mmsteer_transforms_forward, fout)

#     mmsteer_transforms_inverse = calculate_mmster(
#         pos_means=neg_handler.means,
#         pos_covariances=neg_handler.covariances,
#         neg_means=pos_handler.means,
#         neg_covariances=pos_handler.covariances,
#     )
#     with open(f'{output_dir}/mmsteer_inverse_{step}.pickle', 'wb') as fout:
#         pickle.dump(mmsteer_transforms_inverse, fout)


def calculate_casteer(pos_means: dict, neg_means: dict) -> dict:
    result = {}
    for denoising_step in pos_means.keys():
        result[denoising_step] = {}
        for place_in_unet in pos_means[denoising_step].keys():
            result[denoising_step][place_in_unet] = []
            for block_idx in range(len(pos_means[denoising_step][place_in_unet])):
                print(f'Processing step={denoising_step}, block={place_in_unet}, layer={block_idx}')
                steering_vector = (
                    pos_means[denoising_step][place_in_unet][block_idx] -
                    neg_means[denoising_step][place_in_unet][block_idx]
                )
                steering_vector /= torch.linalg.norm(steering_vector, dim=1, keepdim=True)
                result[denoising_step][place_in_unet].append(steering_vector.to(torch.float32).detach().cpu().numpy())
    return result


def fractional_matrix_power_cov(A: np.ndarray, p: float, eps=1e-10):
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, 0)
    mask = (evals >= eps)
    evals = evals[mask]
    evecs = evecs[:, mask]
    return evecs @ np.diag(evals ** p) @ evecs.T


def calculate_mmster(
        pos_means: dict,
        pos_covariances: dict,
        neg_means: dict,
        neg_covariances: dict
) -> dict:
    result = {}
    for denoising_step in pos_means.keys():
        result[denoising_step] = {}
        for place_in_unet in pos_means[denoising_step].keys():
            result[denoising_step][place_in_unet] = []
            for block_idx in range(len(pos_means[denoising_step][place_in_unet])):
                start = time.time()
                
                mu_pos = pos_means[denoising_step][place_in_unet][block_idx]
                sigma_pos = pos_covariances[denoising_step][place_in_unet][block_idx]

                mu_neg = neg_means[denoising_step][place_in_unet][block_idx]
                sigma_neg = neg_covariances[denoising_step][place_in_unet][block_idx]

                
                sigma_neg_half = fractional_matrix_power_cov_torch(sigma_neg, 0.5)
                sigma_neg_minus_half = fractional_matrix_power_cov_torch(sigma_neg, -0.5)
                W = fractional_matrix_power_cov_torch(sigma_neg_half @ sigma_pos @ sigma_neg_half, 0.5)
                W = sigma_neg_minus_half @ W @ sigma_neg_minus_half

                b = (- W @ mu_neg[..., None])[..., 0] + mu_pos

                print(f'Processing step={denoising_step:2}, block={place_in_unet:4}, layer={block_idx:2}: took {time.time() - start:.2f} s '
                      f'|W|_2 = {torch.linalg.norm(W, ord=2, dim=(1, 2))}, |b|_2 = {torch.linalg.norm(b, dim=1)}')

                result[denoising_step][place_in_unet].append((
                    W.to(torch.float32).detach().cpu().numpy(),
                    b.to(torch.float32).detach().cpu().numpy(),
                ))
    return result


def run(args: argparse.Namespace):
    pipe = init_pipeline_for_image_model(args.model)
    pipe.set_progress_bar_config(disable=True)

    device = get_device()


    if args.mode == 'concrete':
        prompts_pos, prompts_neg = get_prompts_concrete(concept_pos=args.concept_pos, 
                                                        concept_neg=args.concept_neg)
    elif args.mode == 'human-related':
        prompts_pos, prompts_neg = get_prompts_human_related(concept_pos=args.concept_pos, 
                                                            concept_neg=args.concept_neg)
    elif args.mode == 'style':
        prompts_pos, prompts_neg = get_prompts_style(concept_pos=args.concept_pos, 
                                                    concept_neg=args.concept_neg)
    elif args.mode == 'file':
        prompts_pos = read_prompt_file(args.prompts_pos_file)
        prompts_neg = read_prompt_file(args.prompts_neg_file)
        


    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_steps = set(map(int, args.checkpoint_steps.split(',')))

    if prompts_pos is not None and prompts_neg is not None:
        gather_stats_for_prompt_pairs(
            pipe=pipe,
            args=args,
            checkpoint_steps=checkpoint_steps,
            device=device,
            prompts_pos=prompts_pos,
            prompts_neg=prompts_neg,
        )
    else:
        gather_stats_for_prompts(
            pipe=pipe,
            prompts=prompts_neg,
            model_type=args.model,
            device=device,
            patch_average=args.patch_average,
            output_dir=args.output_dir,
            normalize_vectors=args.normalize_vectors,
            checkpoint_steps=checkpoint_steps,
            control_mode=args.control_mode,
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True)
    parser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'style', 'file'], default="style")
    parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode], default='attn_output', help='Vector control mode')
    parser.add_argument('--prompts_pos_file', type=str, default=None,
                        help="If --mode is set to 'file', path to the text file containing positive prompts")
    parser.add_argument('--prompts_neg_file', type=str, default=None,
                        help="If --mode is set to 'file', path to the text file containing negative prompts")
    parser.add_argument('--concept_pos', type=str, default="anime")
    parser.add_argument('--concept_neg', type=str, default=None)
    parser.add_argument('--patch_average', action='store_true', help='Average across patches for each prompt before updating statistics')
    parser.add_argument('--normalize_vectors', action='store_true', help='Whether to normalize vectors before computing the statistics')
    parser.add_argument('--output_dir', type=str, default=None, required=True, help='path to saving steering vectors')
    parser.add_argument('--checkpoint_steps', type=str, default='100,500,1000,5000,10000', help='A comma separated list of integers representing steps at which to write checkpoints')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()