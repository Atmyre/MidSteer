#! /usr/bin/env bash
#SBATCH --job-name=midsteer_base_experiment
#SBATCH --partition=h100-camera-train
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_base_experiment-%j.out
#SBATCH --error=slurm_base_experiment-%j.err


set -eoux pipefail

export PYTHONPATH=.

python=../miniconda3/bin/python

model_name=meta-llama/Llama-2-7b-chat-hf
layer_type=self_attn
covariances_dir=./llm_exp/cov/llama-2-7b-chat-hf/$layer_type

topics="horses motorcycles cats dogs"
steering_vectors_dir=./llm_exp/steering_vectors/llama-2-7b-chat-hf/$layer_type


srun $python concept_flipping/estimate_covariances.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --token_aggregation_mode all \
    --num_samples 20000 \
    --output_dir $covariances_dir


srun $python concept_flipping/generate_steering_vectors.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --topics $topics \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir



alpaca_num_samples=1000
alpaca_max_new_tokens=100
alpaca_samples_per_question=1

eval_max_new_tokens=100
eval_samples_per_question=10


results_dir=./llm_exp/results/llama-2-7b-chat-hf/$layer_type

# Iterate over concept pairs
declare -A concept_pairs=(
    ["horses"]="motorcycles"
    ["dogs"]="cats"
)

for source_concept in "${!concept_pairs[@]}"; do
    target_concept="${concept_pairs[$source_concept]}"
    
    results_subdir="$results_dir/${source_concept}_to_${target_concept}"
    mkdir -p "$results_subdir"

    # Define evaluation parameters as arrays
    declare -a eval_params=(
        "--samples_per_question $eval_samples_per_question --max_new_tokens $eval_max_new_tokens --output_dir $results_subdir/eval"
        "--alpaca_eval --alpaca_num_samples $alpaca_num_samples --samples_per_question $alpaca_samples_per_question --max_new_tokens $alpaca_max_new_tokens --output_dir $results_subdir/alpaca"
    )

    for params in "${eval_params[@]}"; do
        srun $python concept_flipping/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $source_concept \
            --strength 0.0 \
            $params

        srun $python concept_flipping/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $source_concept \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type casteer \
            --strength 2.0 \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $params

        srun $python concept_flipping/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $source_concept \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type leace \
            --strength 2.0 \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $params


        i=0
        for strength in 1.0 1.5 2.0 2.5; do
            gpu_id=$((3 + i))
            srun $python concept_flipping/run_with_steering.py \
                --model_name $model_name \
                --layer_type $layer_type \
                --source_concept $source_concept \
                --source_concept_path $steering_vectors_dir/$source_concept.pt \
                --target_concept_path $steering_vectors_dir/$target_concept.pt \
                --steer_type mean_matching \
                --strength $strength \
                --mu_neutral $covariances_dir/means.pt \
                --cov_neutral $covariances_dir/covariances.pt \
                $params
            i=$((i + 1))
        done

    done


    srun $python concept_flipping/llama_scoring.py \
        --concept $source_concept $target_concept \
        --dir $results_subdir/eval

    srun $python concept_flipping/alpaca_scoring.py \
        --dir $results_subdir/alpaca



done



