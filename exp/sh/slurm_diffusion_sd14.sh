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
seed=0

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


additional_steering_params="--model_name $model_name --control_mode $control_mode $intermediate_clipping $renormalize_after_steering --num_images_per_prompt $num_images_per_prompt --seed $seed --file_format JPEG --batch_size 10"
base_dir=./exp/results/$model_name/$SLURM_JOB_NAME

export PYTHONPATH=.

python=../miniconda3/bin/python

covariances_dir=$base_dir/covariances

topics="snoopy mickey"
steering_vectors_dir=$base_dir/steering_vectors
estimate_model_name=$model_name

if [[ $model_name == "sdxl" ]]; then
    estimate_model_name="sdxl-turbo"
elif [[ $model_name == "sana" ]]; then
    estimate_model_name="sana-sprint"
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

#$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
#    --model_name $estimate_model_name \
#    --control_mode $control_mode \
#    --normalize_vectors \
#    --aggregation_mode average \
#    --num_samples 1000 \
#    --output_dir $steering_vectors_dir &
#
#wait


results_dir=$base_dir/evaluation/

# Iterate over concept pairs
declare -A concept_pairs=(
    #["horse"]="motorcycle"
    ["snoopy"]="mickey"
    #["chihuahua"]="muffin"
)


declare -a concepts_to_remove_pairs=(
    "snoopy:snoopy"
    "snoopy:mickey"
    "snoopy:spongebob"
    "snoopy:pikachu"
    "snoopy:dog"
    "snoopy:legislator"
    #"horse:horse"
    #"horse:motorcycle"
    #"horse:cow"
    #"horse:pig"
    #"horse:dog"
    #"horse:legislator"
    #"chihuahua:chihuahua"
    #"chihuahua:muffin"
    #"chihuahua:wolf"
    #"chihuahua:cat"
    #"chihuahua:dog"
    #"chihuahua:legislator"
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
	    --model_name "sd14" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength $strength \
	    --covariances_dir "/home/t50045037/mmsteer/exp/results/sd14/sd14_snoopy_normed/steering_vectors/" \
	    --use_all_diffusion_steps \
            $additional_steering_params \
            erase \
            --concept_path $steering_vectors_dir/$concept_to_remove.pt &
    done



    wait
done


#for pair in "${concepts_to_remove_pairs[@]}"; do
#    IFS=':' read -r concept_to_remove concept_to_generate <<< "$pair"
#    
#    results_subdir="$results_dir/concept_erasure/${concept_to_remove}_${concept_to_generate}"
#
#    $run_cmd $python scripts/diffusion/produce_scores.py \
#        --concept "$concept_to_remove" "$concept_to_generate" \
#        --dir "$results_subdir" &
#done

wait
