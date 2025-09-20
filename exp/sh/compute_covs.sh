#! /usr/bin/env bash
#SBATCH --partition=camera-xlong
#SBATCH --gpus=8
#SBATCH --time=120:00:00
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

topics="horse motorcycle snoopy mickey chihuahua muffin"
steering_vectors_dir=$base_dir/steering_vectors
estimate_model_name=$model_name

if [[ $model_name == "sdxl" ]]; then
    estimate_model_name="sdxl-turbo"
elif [[ $model_name == "sana" ]]; then
    estimate_model_name="sana-sprint"
fi


if [ $num_covariances -gt 0 ]; then
    $run_cmd $python scripts/diffusion/estimate_covariances.py \
        --model_name $estimate_model_name \
        --control_mode $control_mode \
        --aggregation_mode $aggregation_mode \
	--normalize_vectors \
        --num_samples $num_covariances \
        --output_dir $covariances_dir &

    additional_steering_params="$additional_steering_params --covariances_dir $covariances_dir"
fi


$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
    --model_name $estimate_model_name \
    --control_mode $control_mode \
    --topics $topics \
    --normalize_vectors \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir &

wait


