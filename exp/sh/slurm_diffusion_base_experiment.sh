#! /usr/bin/env bash
#SBATCH --partition=camera-xlong
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --output=./exp/logs/slurm-%x-%j.out
#SBATCH --error=./exp/logs/slurm-%x-%j.err
 

set -eoux pipefail


# Check if required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_name> <control_mode> <num_covariances> <aggregation_mode> [strengths] [--intermediate_clipping] [--renormalize_after_steering]"
    echo "Example: $0 sdxl attn_output 20000 all" 
    echo "Example with strengths: $0 sdxl attn_output 20000 all '1.0 2.0 3.0'"
    echo "Example with optional flags: $0 sdxl attn_output 20000 all '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping"
    exit 1
fi

# Parse arguments
model_name=$1
control_mode=$2
num_covariances=$3
aggregation_mode=$4
num_images_per_prompt=10
seed=42

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

intermediate_clipping=""
renormalize_after_steering=""

# Check for optional arguments
for arg in "${@:$start_idx}"; do
    case $arg in
        "--intermediate_clipping")
            intermediate_clipping="--intermediate_clipping"
            ;;
        "--renormalize_after_steering")
            renormalize_after_steering="--renormalize_after_steering"
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


additional_steering_params="--model_name $model_name --control_mode $control_mode $intermediate_clipping $renormalize_after_steering --num_images_per_prompt $num_images_per_prompt --seed $seed"
base_dir=./exp/results/$model_name/$SLURM_JOB_NAME

export PYTHONPATH=.

python=../miniconda3/bin/python

covariances_dir=$base_dir/covariances

topics="horses motorcycles cats dogs"
steering_vectors_dir=$base_dir/steering_vectors

if [ $num_covariances -gt 0 ]; then
    $run_cmd $python scripts/diffusion/estimate_covariances.py \
        --model_name $model_name \
        --control_mode $control_mode \
        --aggregation_mode $aggregation_mode \
        --num_samples $num_covariances \
        --output_dir $covariances_dir &

    additional_steering_params="$additional_steering_params --covariances_dir $covariances_dir"
fi


$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
    --model_name $model_name \
    --control_mode $control_mode \
    --topics $topics \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir &

wait


results_dir=$base_dir/evaluation/

# Iterate over concept pairs
declare -A concept_pairs=(
    ["horse"]="motorcycle"
    ["dog"]="cat"
)

declare -a concepts_to_steer_pairs=(
    "horse:horse"
    "horse:cow"
    "horse:motorcycle"
    "horse:dog"
    "horse:legislator"
    "dog:dog"
    "dog:wolf"
    "dog:cat"
    "dog:horse"
    "dog:legislator"
)



for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"
    
    # Sanitize concept_to_steer for directory name (replace spaces and apostrophes with underscores)
    sanitized_concept=$(echo "$concept_to_steer" | sed 's/[[:space:]'\''"]/_/g')
    results_subdir="$results_dir/concept_translation/${source_concept}_to_${target_concept}__${sanitized_concept}"
    mkdir -p "$results_subdir"

    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept_to_steer" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength $strength \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            $additional_steering_params &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength $strength \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            $additional_steering_params &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/mean_matching-$strength" \
            --steering_method mean_matching \
            --steering_strength $strength \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt \
            $additional_steering_params &
    done

    wait


    $run_cmd $python scripts/diffusion/produce_scores.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir"

done




declare -a concepts_to_remove_pairs=(
    "snoopy:snoopy"
    "snoopy:mickey"
    "snoopy:spongebob"
    "snoopy:pikachu"
    "snoopy:dog"
    "snoopy:legislator"
    "mickey:snoopy"
    "mickey:mickey"
    "mickey:spongebob"
    "mickey:pikachu"
    "mickey:dog"
    "mickey:legislator"
)




for pair in "${concepts_to_remove_pairs[@]}"; do
    IFS=':' read -r concept_to_remove concept_to_generate <<< "$pair"
    
    results_subdir="$results_dir/concept_erasure/${concept_to_remove}_${concept_to_generate}"
    mkdir -p "$results_subdir"


    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept_to_generate" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength $strength \
            erase \
            --concept_path $steering_vectors_dir/$concept_to_remove.pt \
            $additional_steering_params &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength $strength \
            erase \
            --concept_path $steering_vectors_dir/$concept_to_remove.pt \
            $additional_steering_params &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/mean_matching-$strength" \
            --steering_method mean_matching \
            --steering_strength $strength \
            erase \
            --concept_path $steering_vectors_dir/$concept_to_remove.pt \
            $additional_steering_params &
    done

    wait


    $run_cmd $python scripts/diffusion/produce_scores.py \
        --concept "$concept_to_remove" "$concept_to_generate" \
        --dir "$results_subdir"

done