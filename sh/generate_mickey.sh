#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/mickey_retest_no_norm"
seeds="0,42"  # String representing list of seeds, separated by comma
prompt_file="./test_prompts/prompts_for_mickey.txt"

for model in "sdxl"; do


    echo "Forward generating for $model "
    path="$PREFIX/forward/$model/"
    mkdir -p "$path"

    python generate_casteer.py \
        --model $model \
        --prompt_file $prompt_file \
         \
        --seed "$seeds" --output "$path" --not_steer

    for alpha in 1; do
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --steering_vectors ./steering_vectors/mickey_laion_steering_vectors_unnormed.pickle \
            --alpha "${alpha}" --steer_type casteer
    done

    for alpha in 1; do 
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --steering_vectors ./steering_vectors/mickey_laion_mm_steering_vectors_unnormed.pickle \
            --alpha "${alpha}" --steer_type mmsteer
    done
done


prompt_file="./test_prompts/prompts_for_mickey_inverse.txt"

for model in "sdxl"; do


    echo "Inverse generating for $model "
    path="$PREFIX/inverse/$model/"
    mkdir -p "$path"

    python generate_casteer.py \
        --model $model \
        --prompt_file $prompt_file \
         \
        --seed "$seeds" --output "$path" --not_steer

    for beta in 1; do
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --steering_vectors ./steering_vectors/mickey_laion_steering_vectors_unnormed.pickle \
            --beta "${beta}" --steer_type casteer --steer_back
    done

    for alpha in 1; do 
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
             \
            --seed "$seeds" --output "$path" \
            --steering_vectors ./steering_vectors/mickey_laion_mm_inverse_steering_vectors_unnormed.pickle \
            --alpha "${alpha}" --steer_type mmsteer
    done
done
