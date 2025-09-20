#!/usr/bin/env bash

sbatch --gpus=1 --job-name=sd14_snoopy ./exp/sh/compute_steering_vecs.sh sd14 attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
#sbatch --gpus=1 --job-name=sana_cifar ./exp/sh/compute_steering_vecs.sh sana attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
