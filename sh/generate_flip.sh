#!/usr/bin/env bash

set -eoux pipefail

prompt_file="./test_prompts/horse_prompts.txt"
PREFIX="images/horse_motorcycle_10000_test"
seeds="94,378"  # String representing list of seeds, separated by comma

for model in "sdxl"; do


    echo "Forward generating for $model "
    path="$PREFIX/output/$model/"
    mkdir -p "$path"

#     python generate_casteer.py \
#         --model $model \
#         --prompt_file $prompt_file \
#          \
#         --seed "$seeds" --output "$path" --not_steer

#     for beta in 2; do
#         python generate_casteer.py \
#             --model $model \
#             --prompt_file $prompt_file \
#             --control_mode attn_output \
#              \
#             --seed "$seeds" --output "$path" \
#             --casteer_vectors ./ckpt/horse_motorcycle_output/casteer_10000.pickle \
#             --beta "${beta}" --steer_type casteer --steer_back
#     done

#     for alpha in 1; do 
#         python generate_casteer.py \
#             --model $model \
#             --prompt_file $prompt_file \
#             --control_mode attn_output \
#              \
#             --seed "$seeds" --output "$path" \
#             --mmsteer_vectors ./ckpt/horse_motorcycle_output/mmsteer_inverse_10000.pickle \
#             --alpha "${alpha}" --steer_type mmsteer
#     done
    
    for alpha in 0; do
        python generate_casteer.py \
        --model $model \
        --control_mode attn_output \
        --prompt_file "$prompt_file"  \
        --seed "$seeds" --output "$path" \
        --mu_pos ckpt/horse_motorcycle_output/neg_means_10000.pickle \
        --mu_neg ckpt/horse_motorcycle_output/pos_means_10000.pickle \
        --mu_neutral ckpt/all_output/means_99999.pickle \
        --cov ckpt/all_output/covariances_99999.pickle \
        --alpha "${alpha}" \
        --steer_type mean_matching 
    done
done







