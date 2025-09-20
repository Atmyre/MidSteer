#!/usr/bin/env bash

sbatch --gpus=4 --job-name=sd14_snoopy_normed ./exp/sh/slurm_diffusion_sd14.sh sd14 attn_output 10000 average '2.0'
sbatch --gpus=4 --job-name=sd14_snoopy_clip_normed ./exp/sh/slurm_diffusion_sd14.sh sd14 attn_output 10000 average '2.0' --intermediate_clipping
