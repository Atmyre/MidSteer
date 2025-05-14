#! /usr/bin/env bash

set -eoux pipefail


model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn
source_concept=horses
target_concept=motorcycles
max_new_tokens=150
mu_neutral_path=./ckpt/llama2-7b-chat_alpaca_all_self_attn_all/pos_means_1000.pickle
cov_neutral_path=./ckpt/llama2-7b-chat_alpaca_all_self_attn_all/pos_covariances_1000.pickle

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/run_with_steering.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --source_concept $source_concept \
    --target_concept $target_concept \
    --steer_type mean_matching \
    --strength 2.0 \
    --max_new_tokens $max_new_tokens \
    --mu_neutral $mu_neutral_path \
    --cov_neutral $cov_neutral_path
