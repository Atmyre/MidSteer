#! /usr/bin/env bash
#SBATCH --partition=h100-camera-train
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=./llm_exp/logs/slurm-%x-%j.out
#SBATCH --error=./llm_exp/logs/slurm-%x-%j.err


set -eoux pipefail

# Check if required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <layer_type> <num_covariances> <token_aggregation_mode> <max_new_tokens> [--use_alpaca_system_prompt]"
    echo "Example: $0 self_attn 20000 all 100" 
    echo "Example with optional flag: $0 self_attn 20000 all 100 --use_alpaca_system_prompt"
    exit 1
fi

# Parse arguments
layer_type=$1
num_covariances=$2
token_aggregation_mode=$3
max_new_tokens=$4
use_alpaca_system_prompt=""

# Check for optional argument
if [ $# -eq 5 ] && [ "$5" = "--use_alpaca_system_prompt" ]; then
    use_alpaca_system_prompt="--use_alpaca_system_prompt"
fi



base_dir=./llm_exp/results/llama-2-7b-chat-hf/$SLURM_JOB_NAME

export PYTHONPATH=.

python=../miniconda3/bin/python

model_name=meta-llama/Llama-2-7b-chat-hf
covariances_dir=$base_dir/covariances

topics="horses motorcycles cats dogs"
steering_vectors_dir=$base_dir/steering_vectors

if [ $num_covariances -eq 0 ]; then
    srun $python concept_flipping/estimate_covariances.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --token_aggregation_mode $token_aggregation_mode \
        --num_samples 10 \
        --max_new_tokens $max_new_tokens \
        --output_dir $covariances_dir

    additional_params="--identity_cov --zero_mu_neutral"
else
    srun $python concept_flipping/estimate_covariances.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --token_aggregation_mode $token_aggregation_mode \
        --num_samples $num_covariances \
        --max_new_tokens $max_new_tokens \
        --output_dir $covariances_dir

    additional_params=""
fi


srun $python concept_flipping/generate_steering_vectors.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --topics $topics \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir \
    $use_alpaca_system_prompt



alpaca_num_samples=1000
alpaca_max_new_tokens=100
alpaca_samples_per_question=1

eval_max_new_tokens=100
eval_samples_per_question=10


results_dir=$base_dir/evaluation/

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
            $additional_params \
            $use_alpaca_system_prompt \
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
            $additional_params \
            $use_alpaca_system_prompt \
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
            $additional_params \
            $use_alpaca_system_prompt \
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
                $additional_params \
                $use_alpaca_system_prompt \
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



declare -A concepts_to_steer=(
    ["horses"]="cows"
    ["dogs"]="wolves"
)

for source_concept in "${!concept_pairs[@]}"; do
    target_concept="${concept_pairs[$source_concept]}"
    concept_to_steer="${concepts_to_steer[$source_concept]}"
    
    results_subdir="$results_dir/${source_concept}_to_${target_concept}__${concept_to_steer}"
    mkdir -p "$results_subdir"

    params="--samples_per_question $eval_samples_per_question --max_new_tokens $eval_max_new_tokens --output_dir $results_subdir/eval"

    srun $python concept_flipping/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept $concept_to_steer \
        --strength 0.0 \
        $additional_params \
        $use_alpaca_system_prompt \
        $params

    srun $python concept_flipping/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept $concept_to_steer \
        --source_concept_path $steering_vectors_dir/$source_concept.pt \
        --target_concept_path $steering_vectors_dir/$target_concept.pt \
        --steer_type casteer \
        --strength 2.0 \
        --mu_neutral $covariances_dir/means.pt \
        --cov_neutral $covariances_dir/covariances.pt \
        $additional_params \
        $use_alpaca_system_prompt \
        $params

    srun $python concept_flipping/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept $concept_to_steer \
        --source_concept_path $steering_vectors_dir/$source_concept.pt \
        --target_concept_path $steering_vectors_dir/$target_concept.pt \
        --steer_type leace \
        --strength 2.0 \
        --mu_neutral $covariances_dir/means.pt \
        --cov_neutral $covariances_dir/covariances.pt \
        $additional_params \
        $use_alpaca_system_prompt \
        $params


    i=0
    for strength in 1.0 1.5 2.0 2.5; do
        gpu_id=$((3 + i))
        srun $python concept_flipping/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $concept_to_steer \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type mean_matching \
            --strength $strength \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $additional_params \
            $use_alpaca_system_prompt \
            $params
        i=$((i + 1))
    done


    srun $python concept_flipping/llama_scoring.py \
        --concept $source_concept $target_concept $concept_to_steer \
        --dir $results_subdir/eval

done
