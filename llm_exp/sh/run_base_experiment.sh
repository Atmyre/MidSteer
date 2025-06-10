#! /usr/bin/env bash

set -eoux pipefail

model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn
covariances_dir=./llm_exp/cov/llama-2-7b-chat-hf/self_attn

topics="horses motorcycles cats dogs"
steering_vectors_dir=./llm_exp/steering_vectors/llama-2-7b-chat-hf/self_attn


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/estimate_covariances.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --token_aggregation_mode all \
    --num_samples 20000 \
    --output_dir $covariances_dir &


CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python concept_flipping/generate_steering_vectors.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --topics $topics \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir &

wait



eval_tokens=150

results_dir=./llm_exp/results/llama-2-7b-chat-hf/self_attn

# Iterate over concept pairs
declare -A concept_pairs=(
    ["horses"]="motorcycles"
    ["dogs"]="cats"
)

for source_concept in "${!concept_pairs[@]}"; do
    target_concept="${concept_pairs[$source_concept]}"
    
    results_subdir="$results_dir/${source_concept}_to_${target_concept}"
    mkdir -p "$results_subdir"


    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept $source_concept \
        --source_concept_path $steering_vectors_dir/$source_concept.json \
        --target_concept_path $steering_vectors_dir/$target_concept.json \
        --steer_type casteer \
        --strength 2.0 \
        --max_new_tokens $eval_tokens \
        --mu_neutral $covariances_dir/means.pt \
        --cov_neutral $covariances_dir/covariances.pt \
        --output_dir $results_subdir/casteer_2.0 &


    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python concept_flipping/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept $source_concept \
        --source_concept_path $steering_vectors_dir/$source_concept.json \
        --target_concept_path $steering_vectors_dir/$target_concept.json \
        --steer_type leace \
        --strength 2.0 \
        --max_new_tokens $eval_tokens \
        --mu_neutral $covariances_dir/means.pt \
        --cov_neutral $covariances_dir/covariances.pt \
        --output_dir $results_subdir/leace_2.0 &


    i=0
    for strength in 1.0 1.5 2.0 2.5; do
        gpu_id=$((2 + i))
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python concept_flipping/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $source_concept \
            --source_concept_path $steering_vectors_dir/$source_concept.json \
            --target_concept_path $steering_vectors_dir/$target_concept.json \
            --steer_type mean_matching \
            --strength $strength \
            --max_new_tokens $eval_tokens \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            --output_dir $results_subdir/mean_matching_${strength} &
        i=$((i + 1))
    done

    wait

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python concept_flipping/llama_scoring.py \
        --concept $source_concept $target_concept \
        --dir $results_subdir

done



