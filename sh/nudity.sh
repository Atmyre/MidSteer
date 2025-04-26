#!/usr/bin/env bash

set -eoux pipefail

prompt_file="./test_prompts/neutral_prompts.txt"
PREFIX="images/nudity_1000_test"
seeds="73,89"  # String representing list of seeds, separated by comma

for model in "sdxl"; do


    echo "Forward generating for $model "
    path="$PREFIX/output/$model/"
    mkdir -p "$path"

#     python generate_casteer.py \
#         --model $model \
#         --prompt_file $prompt_file \
#          \
#         --seed "$seeds" --output "$path" --not_steer

    for beta in 1 2; do
        python generate_casteer.py \
            --model $model \
            --prompt_file $prompt_file \
            --control_mode attn_output \
             \
            --seed "$seeds" --output "$path" \
            --casteer_vectors ./ckpt/nudity_custom_big_output/casteer_1000.pickle \
            --beta "${beta}" --steer_type casteer --steer_back
    done

#     for alpha in 1; do 
#         python generate_casteer.py \
#             --model $model \
#             --prompt_file $prompt_file \
#             --control_mode attn_output \
#              \
#             --seed "$seeds" --output "$path" \
#             --mmsteer_vectors ./ckpt/nudity_output/mmsteer_inverse_10000.pickle \
#             --alpha "${alpha}" --steer_type mmsteer
#     done
    
#     for alpha in 1; do
#         python generate_casteer.py \
#         --model $model \
#         --control_mode attn_output \
#         --prompt_file "$prompt_file"  \
#         --seed "$seeds" --output "$path" \
#         --casteer_vectors ckpt/nudity_output/casteer_10000.pickle \
#         --steer_type leace \
#         --alpha 1 \
#         --leace_cov ckpt/nudity_output/neg_covariances_10000.pickle \
#         --leace_mean ckpt/nudity_output/neg_means_10000.pickle 
#     done
    
#     for alpha in 0; do
#         python generate_casteer.py \
#         --model $model \
#         --control_mode attn_output \
#         --prompt_file "$prompt_file"  \
#         --seed "$seeds" --output "$path" \
#         --mu_pos ckpt/nudity_custom_big_output/neg_means_1000.pickle \
#         --mu_neg ckpt/nudity_custom_big_output/pos_means_1000.pickle \
#         --mu_neutral ckpt/all_output/means_99999.pickle \
#         --cov ckpt/all_output/covariances_99999.pickle \
#         --alpha "${alpha}" \
#         --steer_type mean_matching 
#     done
    
done







