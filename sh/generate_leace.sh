#!/usr/bin/env bash

# PREFIX="images/removal_10k/output/"
# seed="0,1,2,3,4,5,6,7,8,9"
# prompt_file="test_prompts/mickey_prompts.txt"

prompt_file="./test_prompts/nudity.txt"
PREFIX="images/nudity_removal_10000"
seeds="78,789"  # String representing list of seeds, separated by comma

for model in "sdxl"; do


    echo "Generating LEACE for $model "
    path="$PREFIX/mickey/inverse/$model/"
    mkdir -p "$path"

#     for beta in 1; do
#         CUDA_VISIBLE_DEVICES=5 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --casteer_vectors ckpt/mickey_output/casteer_10000.pickle --steer_type leace --leace_cov ckpt/mickey_output/neg_covariances_10000.pickle --leace_mean ckpt/mickey_output/neg_means_10000.pickle &
#     done

     python generate_casteer.py \
        --model $model \
        --control_mode attn_output \
        --prompt_file "$prompt_file"  \
        --seed "$seeds" --output "$path" \
        --casteer_vectors ckpt/nudity_output/casteer_10000.pickle \
        --steer_type leace \
        --leace_cov ckpt/nudity_output/neg_covariances_10000.pickle \
        --leace_mean ckpt/nudity_output/neg_means_10000.pickle &

    wait
done