#!/usr/bin/env bash

set -eoux pipefail

PREFIX="images/mickey/"
seed="0,1,2,3,4,5,6,7,8,9"
prompt_file="test_prompts/snoopy_prompts.txt"

for model in "sdxl"; do


    echo "Inverse generating for $model "
    path="$PREFIX/snoopy/inverse/$model/"
    mkdir -p "$path"

#     CUDA_VISIBLE_DEVICES=2 python generate_casteer.py --model $model --prompt_file "$prompt_file" --seed "$seed" --output "$path" --not_steer &

    for beta in 1; do
        CUDA_VISIBLE_DEVICES=3 python generate_casteer.py --model $model --control_mode attn_output --prompt_file "$prompt_file"  --seed "$seed" --output "$path" --casteer_vectors ckpt/mickey_output/casteer_1000.pickle --beta "${beta}" --steer_type casteer --steer_back &
    done

#     for alpha in 1; do
#         CUDA_VISIBLE_DEVICES=0 python generate_casteer.py \
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