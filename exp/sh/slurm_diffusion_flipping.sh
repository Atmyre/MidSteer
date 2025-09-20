#! /usr/bin/env bash
#SBATCH --partition=camera-xlong
#SBATCH --gpus=8
#SBATCH --time=120:00:00
#SBATCH --output=./exp/logs/slurm-%x-%j.out
#SBATCH --error=./exp/logs/slurm-%x-%j.err
#$ -cwd
#$ -j y
#$ -pe smp 32
#$ -l h_rt=120:00:00
#$ -l h_vmem=7.5G
#$ -l gpu=4
#$ -l cluster=apocrita

set -eoux pipefail


# Check if required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_name> <control_mode> <num_covariances> <aggregation_mode> <dataset_type> [strengths] [--intermediate_clipping] [--renormalize_after_steering]"
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
dataset_type=$5
num_images_per_prompt=10
seed=0

# Set default strengths
default_strengths="1.0 1.5 2.0 2.5 3.0 4.0 5.0"

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



if [ -n "${SGE_ROOT:-}" ]; then
    export LOCAL_SCHEDULER=1
    export OUTPUT_PREFIX=/data/scratch/$USER/mmsteer/
    
    if [ -n "${SGE_HGR_gpu:-}" ]; then
        export NUM_GPUS_FOR_LOCAL_SCHEDULER=$(echo $SGE_HGR_gpu | wc -w)
    else
        export NUM_GPUS_FOR_LOCAL_SCHEDULER=1
    fi
elif [ -n "${SLURM_JOB_NAME:-}" ]; then
    export JOB_NAME=$SLURM_JOB_NAME
    export JOB_ID=$SLURM_JOB_ID
    export OUTPUT_PREFIX=.
else
    echo "Error: No job manager found"
    exit 1
fi



if [ -n "${LOCAL_SCHEDULER:-}" ]; then
    export LOCK_FILE="./exp/locks/gpu_pool-${JOB_NAME}-${JOB_ID}.lock"
    rm -rf $LOCK_FILE

    source ./exp/sh/local_scheduler.sh

    for i in $(seq 0 $(($NUM_GPUS_FOR_LOCAL_SCHEDULER-1)) ); do
        release_gpu $i
    done

    run_cmd="run_command_with_params_on_gpu"
else
    run_cmd="srun --gpus=1 -N1 --exclusive"
fi


additional_steering_params="--model_name $model_name --control_mode $control_mode $intermediate_clipping $renormalize_after_steering --num_images_per_prompt $num_images_per_prompt --seed $seed --file_format JPEG --batch_size 10"
base_dir=$OUTPUT_PREFIX/exp/results/$model_name/$JOB_NAME

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
    --dataset_type $dataset_type \
    --control_mode $control_mode \
    --topics $topics \
    --normalize_vectors \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir $steering_vectors_dir &

wait


results_dir=$base_dir/evaluation/

# Iterate over concept pairs
declare -A concept_pairs=(
    ["horse"]="motorcycle"
    ["snoopy"]="mickey"
    ["chihuahua"]="muffin"
)

declare -a concepts_to_steer_pairs=(
    "horse:horse"
    "horse:motorcycle"
    "horse:cow"
    "horse:pig"
    "horse:dog"
    "horse:legislator"
    "snoopy:snoopy"
    "snoopy:mickey"
    "snoopy:Pikachu"
    "snoopy:Spongebob"
    "snoopy:dog"
    "snoopy:legislator"
    "chihuahua:chihuahua"
    "chihuahua:muffin"
    "chihuahua:wolf"
    "chihuahua:cat"
    "chihuahua:dog"
    "chihuahua:legislator"
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
            $additional_steering_params \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength $strength \
            $additional_steering_params \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt &
    done

    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/mean_matching-$strength" \
            --steering_method mean_matching \
            --steering_strength $strength \
            $additional_steering_params \
            translate \
            --source_concept_path $steering_vectors_dir/$source_concept.pt \
            --target_concept_path $steering_vectors_dir/$target_concept.pt &
    done

done

wait


#for pair in "${concepts_to_steer_pairs[@]}"; do
#    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
#    target_concept="${concept_pairs[$source_concept]}"
#    
#    # Sanitize concept_to_steer for directory name (replace spaces and apostrophes with underscores)
#    sanitized_concept=$(echo "$concept_to_steer" | sed 's/[[:space:]'\''"]/_/g')
#    results_subdir="$results_dir/concept_translation/${source_concept}_to_${target_concept}__${sanitized_concept}"
#
#    $run_cmd $python scripts/diffusion/produce_scores.py \
#        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
#        --dir "$results_subdir" &
#done

#wait



