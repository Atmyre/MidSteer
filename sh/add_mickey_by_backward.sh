#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/adding/"
seed="78,42"
prompt_file="test_prompts/prompts_for_mickey.txt"

for model in "sdxl"; do


    echo "Inverse generating for $model "
    path="$PREFIX/mickey/$model/"
    mkdir -p "$path"

#     CUDA_VISIBLE_DEVICES=0 python generate_casteer.py --model $model --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --not_steer

    for alpha in 1; do 
        CUDA_VISIBLE_DEVICES=1 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --mmsteer_vectors ckpt/mickey_output_filtered/mmsteer_forward_1000.pickle --alpha "${alpha}" --steer_type mmsteer
    done

done