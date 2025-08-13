#! /usr/bin/env bash
#SBATCH --partition=camera-xlong
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --output=./exp/logs/slurm-%x-%j.out
#SBATCH --error=./exp/logs/slurm-%x-%j.err


set -eoux pipefail


# Check if required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <layer_type> <num_covariances> <token_aggregation_mode> <max_new_tokens> [strengths] [--mm_normalize_centers] [--intermediate_clipping] [--renormalize_after_steering] [--zero_mu_neutral]"
    echo "Example: $0 self_attn 20000 all 100" 
    echo "Example with strengths: $0 self_attn 20000 all 100 '1.0 2.0 3.0'"
    echo "Example with optional flags: $0 self_attn 20000 all 100 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --mm_normalize_centers --intermediate_clipping"
    exit 1
fi

# Parse arguments
layer_type=$1
num_covariances=$2
token_aggregation_mode=$3
max_new_tokens=$4

# Set default strengths
default_strengths="1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"

# Check if 5th argument is provided and doesn't start with --
if [ $# -gt 4 ] && [[ "$5" != --* ]]; then
    strengths="$5"
    start_idx=6
else
    strengths="$default_strengths"
    start_idx=5
fi

mm_normalize_centers=""
intermediate_clipping=""
renormalize_after_steering=""
zero_mu_neutral=""

# Check for optional arguments
for arg in "${@:$start_idx}"; do
    case $arg in
        "--mm_normalize_centers")
            mm_normalize_centers="--mm_normalize_centers"
            ;;
        "--intermediate_clipping")
            intermediate_clipping="--intermediate_clipping"
            ;;
        "--renormalize_after_steering")
            renormalize_after_steering="--renormalize_after_steering"
            ;;
        "--zero_mu_neutral")
            zero_mu_neutral="--zero_mu_neutral"
            ;;
    esac
done

# Define run_cmd based on NO_SLURM environment variable
if [ -n "${NO_SLURM:-}" ]; then
    run_cmd=""
    export CUDA_VISIBLE_DEVICES=0
else
    run_cmd="srun --gpus=1 -N1 --exclusive"
fi


additional_steering_params="$mm_normalize_centers $intermediate_clipping $renormalize_after_steering $zero_mu_neutral"
base_dir=./exp/results/llama-2-7b-chat-hf/$SLURM_JOB_NAME

export PYTHONPATH=.

python=../miniconda3/bin/python

model_name=meta-llama/Llama-2-7b-chat-hf
covariances_dir=$base_dir/covariances

topics="horses motorcycles cats dogs"
steering_vectors_dir=$base_dir/steering_vectors

if [ $num_covariances -eq 0 ]; then
    $run_cmd $python scripts/llm/estimate_covariances.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --token_aggregation_mode $token_aggregation_mode \
        --num_samples 10 \
        --max_new_tokens $max_new_tokens \
        --output_dir $covariances_dir &

    additional_steering_params="$additional_steering_params --identity_cov"
    if [ -z "$zero_mu_neutral" ]; then
        additional_steering_params="$additional_steering_params --zero_mu_neutral"
    fi
else
    $run_cmd $python scripts/llm/estimate_covariances.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --token_aggregation_mode $token_aggregation_mode \
        --num_samples $num_covariances \
        --max_new_tokens $max_new_tokens \
        --output_dir $covariances_dir &

fi


$run_cmd $python scripts/llm/generate_steering_vectors.py \
    --model_name $model_name \
    --layer_type $layer_type \
    --topics $topics \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir &

wait



consistency_num_samples=1000
consistency_max_new_tokens=100
consistency_samples_per_question=1

concept_max_new_tokens=100
concept_samples_per_question=10


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
        "--dataset_type template --samples_per_question $concept_samples_per_question --max_new_tokens $concept_max_new_tokens --output_dir $results_subdir/eval"
        "--dataset_type alpaca --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/alpaca"
        "--dataset_type mmlu --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/mmlu"
    )

    for params in "${eval_params[@]}"; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $source_concept \
            --strength 0.0 \
            $additional_steering_params \
            $params &

        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name $model_name \
                --layer_type $layer_type \
                --source_concept $source_concept \
                --source_concept_path $steering_vectors_dir/$source_concept.pt \
                --target_concept_path $steering_vectors_dir/$target_concept.pt \
                --steer_type casteer \
                --strength $strength \
                --mu_neutral $covariances_dir/means.pt \
                --cov_neutral $covariances_dir/covariances.pt \
                $additional_steering_params \
                $params &
        done

        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name $model_name \
                --layer_type $layer_type \
                --source_concept $source_concept \
                --source_concept_path $steering_vectors_dir/$source_concept.pt \
                --target_concept_path $steering_vectors_dir/$target_concept.pt \
                --steer_type leace \
                --strength $strength \
                --mu_neutral $covariances_dir/means.pt \
                --cov_neutral $covariances_dir/covariances.pt \
                $additional_steering_params \
                $params &
        done

        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name $model_name \
                --layer_type $layer_type \
                --source_concept $source_concept \
                --source_concept_path $steering_vectors_dir/$source_concept.pt \
                --target_concept_path $steering_vectors_dir/$target_concept.pt \
                --steer_type mean_matching \
                --strength $strength \
                --mu_neutral $covariances_dir/means.pt \
                --cov_neutral $covariances_dir/covariances.pt \
                $additional_steering_params \
                $params &
        done

        wait
    done


    $run_cmd $python scripts/llm/concept_scoring.py \
        --concept $source_concept $target_concept \
        --dir "$results_subdir/eval"

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/alpaca"

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/mmlu"

done



declare -a concepts_to_steer_pairs=(
    "horses:cows"
    "horses:motorcycles"
    "horses:knight's riding mammal"
    "horses:large equine"
    "dogs:wolves"
    "dogs:cats"
    "dogs:man's best friend"
    "dogs:domesticated canine"
)

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"
    
    # Sanitize concept_to_steer for directory name (replace spaces and apostrophes with underscores)
    sanitized_concept=$(echo "$concept_to_steer" | sed 's/[[:space:]'\''"]/_/g')
    results_subdir="$results_dir/${source_concept}_to_${target_concept}__${sanitized_concept}"
    mkdir -p "$results_subdir"

    declare -a concept_params=(--dataset_type template --samples_per_question $concept_samples_per_question --max_new_tokens $concept_max_new_tokens --output_dir $results_subdir/eval)

    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name $model_name \
        --layer_type $layer_type \
        --source_concept "$concept_to_steer" \
        --strength 0.0 \
        $additional_steering_params \
        "${concept_params[@]}" &

    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept "$concept_to_steer" \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type casteer \
            --strength $strength \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $additional_steering_params \
            "${concept_params[@]}" &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept "$concept_to_steer" \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type leace \
            --strength $strength \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $additional_steering_params \
            "${concept_params[@]}" &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept "$concept_to_steer" \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            --steer_type mean_matching \
            --strength $strength \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $additional_steering_params \
            "${concept_params[@]}" &
    done

    wait

    $run_cmd $python scripts/llm/concept_scoring.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir/eval"

done
