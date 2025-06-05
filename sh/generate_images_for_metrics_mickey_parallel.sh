#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/snoopy_sd14/"
seed="0"
prompt_file="test_prompts/mickey_prompts.txt"

for model in "sd14"; do


    echo "Inverse generating for $model "
    path="$PREFIX/mickey/inverse/$model/"
    mkdir -p "$path"

#     CUDA_VISIBLE_DEVICES=0 python generate_casteer.py --model $model --prompt_file "$prompt_file" --seed "$seed" --output "$path/orig/" --not_steer --num_images_per_prompt 10 &

    for beta in 2.5 3; do
        CUDA_VISIBLE_DEVICES=7 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path/50_up/" \
        --mu_pos ckpt/sd14_snoopy_imagenet_comma/pos_means_50.pickle \
        --mu_neg ckpt/sd14_snoopy_imagenet_comma/neg_means_50.pickle \
        --beta "${beta}" --steer_type casteer --steer_back \
        --steer_only_up \
        --num_images_per_prompt 10 &
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