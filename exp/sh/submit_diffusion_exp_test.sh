#!/usr/bin/env bash

sbatch --gpus=4 --job-name=test ./exp/sh/slurm_diffusion_test.sh sdxl attn_output 50000 average '2.0'
