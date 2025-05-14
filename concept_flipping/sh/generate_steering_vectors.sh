#! /usr/bin/env bash

set -eoux pipefail


model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/generate_steering_vectors.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --topics horses motorcycles \
    --token_aggregation_mode last \
    --last_token_offset -1