#! /usr/bin/env bash

set -eoux pipefail

model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn
token_aggregation_mode=all
num_samples=1000


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/estimate_covariances.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --token_aggregation_mode $token_aggregation_mode \
    --num_samples $num_samples
