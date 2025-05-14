#! /usr/bin/env bash

set -eoux pipefail


model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn
source_concept=horses
target_concept=motorcycles
max_new_tokens=150
mu_neutral_path=./concept_flipping/cov/meta-llama_Llama-2-7b-chat-hf_self_attn_means_10000.pt
cov_neutral_path=./concept_flipping/cov/meta-llama_Llama-2-7b-chat-hf_self_attn_covariances_10000.pt

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/run_with_steering.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --source_concept $source_concept \
    --target_concept $target_concept \
    --steer_type casteer \
    --strength 2.0 \
    --max_new_tokens $max_new_tokens \
    --mu_neutral $mu_neutral_path \
    --cov_neutral $cov_neutral_path &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python concept_flipping/run_with_steering.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --source_concept $source_concept \
    --target_concept $target_concept \
    --steer_type leace \
    --strength 2.0 \
    --max_new_tokens $max_new_tokens \
    --mu_neutral $mu_neutral_path \
    --cov_neutral $cov_neutral_path &

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python concept_flipping/run_with_steering.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --source_concept $source_concept \
    --target_concept $target_concept \
    --steer_type mean_matching \
    --strength 1.5 \
    --max_new_tokens $max_new_tokens \
    --mu_neutral $mu_neutral_path \
    --cov_neutral $cov_neutral_path &

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python concept_flipping/run_with_steering.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --source_concept $source_concept \
    --target_concept $target_concept \
    --steer_type mean_matching \
    --strength 2.0 \
    --max_new_tokens $max_new_tokens \
    --mu_neutral $mu_neutral_path \
    --cov_neutral $cov_neutral_path &

wait
