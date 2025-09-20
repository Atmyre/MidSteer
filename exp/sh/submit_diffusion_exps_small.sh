#!/usr/bin/env bash

sbatch --gpus=1 --job-name=midsteer_diffusion_sdxl_20k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 20000 average '1.0 2.0 3.0'
