#!/usr/bin/env bash

sbatch --gpus=4 --job-name=sd14_snoopy_normed ./exp/sh/generate_metrics.sh sd14 attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
