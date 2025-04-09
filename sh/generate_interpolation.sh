#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/interp_horse_motorcycle"
seeds="21"  # String representing list of seeds, separated by comma

for model in "sdxl"; do
    if [[ $model = 'sdxl' ]]; then
        num_denoising_steps=30
    else
        num_denoising_steps=1
    fi

    echo "Forward generating for $model $num_denoising_steps"
    path="$PREFIX/forward/$model/"
    mkdir -p "$path"

    python generate_casteer.py \
        --model $model \
        --prompt_file ./test_prompts/horse_prompts.txt \
        --num_denoising_steps $num_denoising_steps \
        --seed "$seeds" --output "$path" --not_steer

    for alpha in $(seq 3 3 18); do
        python generate_casteer.py \
            --model $model \
            --prompt_file ./test_prompts/horse_prompts.txt \
            --num_denoising_steps $num_denoising_steps \
            --seed "$seeds" --output "$path" \
            --steering_vectors steering_vectors/horse_to_motorcycle_laion_steering_vectors.pickle \
            --alpha "${alpha}" --steer_type casteer
    done

    for alpha in $(seq 0.5 0.25 2.0); do 
        python generate_casteer.py \
            --model $model \
            --prompt_file ./test_prompts/horse_prompts.txt \
            --num_denoising_steps $num_denoising_steps \
            --seed "$seeds" --output "$path" \
            --steering_vectors steering_vectors/horse_to_motorcycle_laion_mm_steering_vectors.pickle \
            --alpha "${alpha}" --steer_type mmsteer
    done
done
