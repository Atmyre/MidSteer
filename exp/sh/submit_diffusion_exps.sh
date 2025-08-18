#!/usr/bin/env bash

sbatch --gpus=4 --job-name=midsteer_diffusion_test ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 10000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
