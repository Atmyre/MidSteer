#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/snoopy_sdxl/"
seed="0,1"
prompt_file="test_prompts/mickey_prompts.txt"

for model in "sdxl"; do


    echo "Inverse generating for $model "
    path="$PREFIX/mickey/inverse/$model/"
    mkdir -p "$path"

#     CUDA_VISIBLE_DEVICES=1 python generate_casteer.py --model $model --prompt_file "$prompt_file" --seed "$seed" --output "$path/orig/" --not_steer --num_images_per_prompt 5 &

    for beta in 2; do
        CUDA_VISIBLE_DEVICES=5 python generate_casteer.py --model $model --control_mode attn_key_value --prompt_file "$prompt_file"  --seed "$seed" --output "$path/50_attn_key_value_n/" \
        --mu_pos ckpt/sdxl_snoopy_key_value/pos_means_50.pickle \
        --mu_neg ckpt/sdxl_snoopy_key_value/neg_means_50.pickle \
        --beta "${beta}" --steer_type casteer --steer_back \
        --num_images_per_prompt 5 &
    done

#     for alpha in 1; do
#         CUDA_VISIBLE_DEVICES=1 python generate_casteer.py \
#             --model $model \
#             --control_mode attn_output \
#             --prompt_file "$prompt_file"  \
#             --seed "$seed" --output "$path" \
#             --casteer_vectors ckpt/mickey_output/casteer_1000.pickle \
#             --steer_type leace \
#             --alpha "${alpha}" \
#             --leace_cov ckpt/all_output/covariances_99999.pickle \
#             --leace_mean ckpt/all_output/means_99999.pickle &
#     done
    
    wait
done