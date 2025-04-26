#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/adding/"
seed="78,42"
prompt_file="test_prompts/prompts_for_mickey_inverse.txt"

for model in "sdxl"; do


    echo "Forward generating for $model "
    path="$PREFIX/mickey/$model/"
    mkdir -p "$path"

#     CUDA_VISIBLE_DEVICES=3 python generate_casteer.py --model $model --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --not_steer

    for beta in 1; do
        CUDA_VISIBLE_DEVICES=4 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --casteer_vectors steering_main_evector.pickle --beta "${beta}" --steer_type casteer --steer_back
    done

#     for alpha in 1; do 
#         CUDA_VISIBLE_DEVICES=1 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --mmsteer_vectors ckpt/mickey_output/mmsteer_forward_10000.pickle --alpha "${alpha}" --steer_type mmsteer &
#     done

    wait
done