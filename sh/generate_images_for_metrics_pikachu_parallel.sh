#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/removal/"
seed="0,1,2,3,4,5,6,7,8,9"
prompt_file="test_prompts/pikachu_prompts.txt"

for model in "sdxl"; do
    if [[ $model = 'sdxl' ]]; then
        num_denoising_steps=30
    else
        num_denoising_steps=1
    fi

    echo "Inverse generating for $model $num_denoising_steps"
    path="$PREFIX/pikachu/inverse/$model/"
    mkdir -p "$path"

    CUDA_VISIBLE_DEVICES=4 python generate_casteer.py --model $model --prompt_file "$prompt_file" --num_denoising_steps $num_denoising_steps --seed "$seed" --output "$path" --not_steer &

    for beta in 1; do
        CUDA_VISIBLE_DEVICES=5 python generate_casteer.py --model $model --prompt_file "$prompt_file" --num_denoising_steps $num_denoising_steps --seed "$seed" --output "$path" --steering_vectors ckpt/mickey_norm/casteer_33715.pickle --beta "${beta}" --steer_type casteer --steer_back &
    done

    for alpha in 1; do 
        CUDA_VISIBLE_DEVICES=6 python generate_casteer.py --model $model --prompt_file "$prompt_file" --num_denoising_steps $num_denoising_steps --seed "$seed" --output "$path" --steering_vectors ckpt/mickey_norm/mmsteer_inverse_33715.pickle --alpha "${alpha}" --steer_type mmsteer &
    done

    wait
done