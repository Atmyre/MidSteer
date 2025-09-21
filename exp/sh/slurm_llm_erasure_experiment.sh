#! /usr/bin/env bash
#SBATCH --partition=camera-xlong
#SBATCH --gpus=4
#SBATCH --time=48:00:00
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


additional_steering_params="$mm_normalize_centers $intermediate_clipping $renormalize_after_steering $zero_mu_neutral"
base_dir=$OUTPUT_PREFIX/exp/results/llama-2-7b-chat-hf/$JOB_NAME

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

for concept_to_delete in "${!concept_pairs[@]}"; do
    
    results_subdir="$results_dir/${concept_to_delete}"
    mkdir -p "$results_subdir"

    # Define evaluation parameters as arrays
    declare -a eval_params=(
        "--dataset_type alpaca --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/alpaca"
        "--dataset_type mmlu --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/mmlu"
    )

    for params in "${eval_params[@]}"; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name $model_name \
            --layer_type $layer_type \
            --source_concept $concept_to_delete \
            --strength 0.0 \
            $additional_steering_params \
            $params &

        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name $model_name \
                --layer_type $layer_type \
                --source_concept $concept_to_delete \
                --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
                --target_concept_path $covariances_dir/means.pt \
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
                --source_concept $concept_to_delete \
                --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
                --target_concept_path $covariances_dir/means.pt \
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
                --source_concept $concept_to_delete \
                --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
                --target_concept_path $covariances_dir/means.pt \
                --steer_type mean_matching \
                --strength $strength \
                --mu_neutral $covariances_dir/means.pt \
                --cov_neutral $covariances_dir/covariances.pt \
                $additional_steering_params \
                $params &
        done

    done
done

wait

for concept_to_delete in "${!concept_pairs[@]}"; do
    results_subdir="$results_dir/${concept_to_delete}"

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/alpaca" &

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/mmlu" &

done

wait



declare -a concepts_to_steer_pairs=(
    "horses:horses"
    "horses:cows"
    "horses:motorcycles"
    "horses:knight's riding mammal"
    "horses:large equine"
    "dogs:dogs"
    "dogs:wolves"
    "dogs:cats"
    "dogs:man's best friend"
    "dogs:domesticated canine"
)

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r concept_to_delete concept_to_steer <<< "$pair"
    
    # Sanitize concept_to_steer for directory name (replace spaces and apostrophes with underscores)
    sanitized_concept=$(echo "$concept_to_steer" | sed 's/[[:space:]'\''"]/_/g')
    results_subdir="$results_dir/${concept_to_delete}_${sanitized_concept}"
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
            --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
            --target_concept_path $covariances_dir/means.pt \
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
            --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
            --target_concept_path $covariances_dir/means.pt \
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
            --source_concept_path $steering_vectors_dir/$concept_to_delete.pt \
            --target_concept_path $covariances_dir/means.pt \
            --steer_type mean_matching \
            --strength $strength \
            --mu_neutral $covariances_dir/means.pt \
            --cov_neutral $covariances_dir/covariances.pt \
            $additional_steering_params \
            "${concept_params[@]}" &
    done
done

wait


for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r concept_to_delete concept_to_steer <<< "$pair"
    
    # Sanitize concept_to_steer for directory name (replace spaces and apostrophes with underscores)
    sanitized_concept=$(echo "$concept_to_steer" | sed 's/[[:space:]'\''"]/_/g')
    results_subdir="$results_dir/${concept_to_delete}_${sanitized_concept}"

    $run_cmd $python scripts/llm/concept_scoring.py \
        --concept "$concept_to_delete" "$concept_to_steer" \
        --dir "$results_subdir/eval" &

done

wait