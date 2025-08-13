import argparse
from collections import defaultdict
import os
from typing import Any

import PIL
import torch

from core.pickle import unpickle
from core.prompts import read_prompt_file
from core.controller import VectorControl, register_vector_controls
from core.utils import get_device, init_pipeline_for_image_model, run_image_model


class Wobbler(VectorControl):
    def __init__(
            self,
            concept_covariances: dict | None,
            wobble_seed: int,
            wobble_only_up: bool,
            wobble_strength: float,
            device: Any,
    ):
        super().__init__()

        self.device = device
        if concept_covariances:
            self._concept_covariances = defaultdict(lambda: defaultdict(list))
            for num_steer in concept_covariances:
                for place_in_unet in concept_covariances[num_steer]:
                    for block_idx in range(len(concept_covariances[num_steer][place_in_unet])):
                        sigma = concept_covariances[num_steer][place_in_unet][block_idx]
                        sigma = torch.tensor(sigma).float().to(self.device)
                        self._concept_covariances[num_steer][place_in_unet].append(sigma)
        else:
            self._concept_covariances = None

        self._wobble_seed = wobble_seed
        self._wobble_only_up = wobble_only_up
        self._wobble_strength = wobble_strength
        self.reset(0)

    def reset(self, iter: int):
        super().reset()
        self._wobble_vectors = defaultdict(lambda: defaultdict(list))
        torch.manual_seed(self._wobble_seed + iter)

    def forward(self, vector, diffusion_step, place_in_unet, block_index):
        diffusion_step = 0
        if place_in_unet in ['up', 'mid'] or (place_in_unet == 'down' and not self._wobble_only_up): 
            if len(self._wobble_vectors[diffusion_step][place_in_unet]) > block_index:
                dh = self._wobble_vectors[diffusion_step][place_in_unet][block_index]
            else:
                if self._concept_covariances is not None:
                    sigma = self._concept_covariances[diffusion_step][place_in_unet][block_index]
                else:
                    sigma = torch.eye(n=vector.shape[-1], device=self.device, dtype=torch.float32)
                mu = torch.zeros(size=(vector.shape[-1],), device=self.device, dtype=torch.float32)
                mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
                dh = self._wobble_strength * mvn.sample((1, 1))
                self._wobble_vectors[diffusion_step][place_in_unet].append(dh)
            return vector + dh.half()


def run(
        model: str,
        prompts: list[str],
        seeds: list[int],
        concept_covariances: dict | None,
        wobble_seed: int,
        num_wobbles: int,
        wobble_only_up: bool,
        wobble_strength: float,
        output_dir: str,
):
    pipe = init_pipeline_for_image_model(model)
    device = get_device()


    controller = Wobbler(
        concept_covariances=concept_covariances,
        wobble_seed=wobble_seed,
        wobble_only_up=wobble_only_up,
        wobble_strength=wobble_strength,
        device=device,
    )
    
    register_vector_controls(pipe.unet, controller)

    for prompt in prompts:
        for seed in seeds:

            controller.active = False
            path = f'{output_dir}/{prompt}/{seed}/orig.png'

            if not os.path.exists(path):
                print(f'Generating original for prompt={prompt}, seed={seed}')
                image = run_image_model(model, pipe, prompt, seed, device=device)[0]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                image.save(path)
            else:
                print(f'{path} already exists, skipping!')

            controller.active = True
            for i in range(num_wobbles):
                controller.reset(i)
                path = f'{output_dir}/{prompt}/{seed}/wobble_{i}.png'
                if os.path.exists(path):
                    print(f'{path} already exists, skipping!')
                    continue
                
                print(f'Generating {i}-th wobble for prompt={prompt}, seed={seed}')
                image = run_image_model(model, pipe, prompt, seed, device=device)[0]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                image.save(path)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sdxl")
    parser.add_argument('--prompt_file', type=str, required=True, help="Path to text file with prompts, one per line.")
    parser.add_argument('--seeds', type=str, default="0", help="Comma-separated list of seeds to use for generation.")
    parser.add_argument('--concept_covariances', type=str, default=None, help='path to covariances of the target concept')
    parser.add_argument('--wobble_seed', default=0, type=int, help='Seed to use for random sampling')
    parser.add_argument('--num_wobbles', type=int, default=10, help='Number of random samples to generate')
    parser.add_argument('--wobble_only_up', action='store_true', help='Only pertrubate the up layers')
    parser.add_argument('--wobble_strength', type=float, default=1, help='Strength of the pertrubation')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output path to image or directory'
    )
    args = parser.parse_args()


    run(
        model=args.model,
        prompts=read_prompt_file(args.prompt_file),
        seeds=list(map(int, args.seeds.split(','))),
        concept_covariances=unpickle(args.concept_covariances),
        wobble_seed=args.wobble_seed,
        num_wobbles=args.num_wobbles,
        wobble_only_up=args.wobble_only_up,
        wobble_strength=args.wobble_strength,
        output_dir=args.output_dir,
    )
    


if __name__ == "__main__":
    main()