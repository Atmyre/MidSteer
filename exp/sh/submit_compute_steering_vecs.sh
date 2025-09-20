#!/usr/bin/env bash

sbatch --gpus=1 --job-name=sana-i2p ./exp/sh/compute_steering_vecs_sana.sh sana attn_output 50000 average '2.0'
