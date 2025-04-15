#!/usr/bin/env bash

set -eoux pipefail

prompt_file="./test_prompts/horse_prompts.txt"
PREFIX="images/interp_horse_to_motorcycle_norm_28205_notrenorm_cls0"
seeds="78"  # String representing list of seeds, separated by comma

for model in "sdxl"; do


    echo "Forward generating for $model "
    path="$PREFIX/forward/$model/"
    mkdir -p "$path"

    python generate_casteer.py \
        --model $model \
        --prompt_file $prompt_file \
         \
        --seed "$seeds" --output "$path" --not_steer

    for alpha in $(seq 3 3 15); do
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --casteer_vectors ./ckpt/horse_to_motorcycle_norm/casteer_28205.pickle \
            --alpha "${alpha}" --steer_type casteer
    done

    for alpha in $(seq 0.5 0.25 2.0); do 
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --mmsteer_vectors ./ckpt/horse_to_motorcycle_norm/mmsteer_forward_28205.pickle \
            --alpha "${alpha}" --steer_type mmsteer
    done
done
