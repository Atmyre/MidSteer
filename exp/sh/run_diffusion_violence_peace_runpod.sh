#!/usr/bin/env bash
# RunPod script for diffusion violence→peace concept switching experiment
# Usage: bash exp/sh/run_diffusion_violence_peace_runpod.sh <sdxl|sana> [strengths]
#
# Runs on both SDXL (via sdxl-turbo estimation) and SANA (via sana-sprint estimation)

set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

# ============================================================
# Configuration
# ============================================================
model_name="${1:?Usage: $0 <sdxl|sana> [strengths]}"
control_mode="attn_output"
num_covariances=50000
aggregation_mode="all"
num_images_per_prompt=10
seed=42
strengths="${2:-1.0 2.0 3.0 4.0 5.0}"
python=python

# Model mapping for estimation
estimate_model_name=$model_name
if [[ $model_name == "sdxl" ]]; then
    estimate_model_name="sdxl-turbo"
elif [[ $model_name == "sana" ]]; then
    estimate_model_name="sana-sprint"
fi

# Install correct diffusers
if [[ "$model_name" == "sana"* ]]; then
    if ! python -c "from diffusers import SanaSprintPipeline" 2>/dev/null; then
        echo "Installing diffusers with SANA support..."
        pip install -q 'diffusers>=0.32.0'
    fi
fi

JOB_NAME="violence_peace"
base_dir="/root/midsteer/exp/results/$model_name/$JOB_NAME"

# ============================================================
# GPU scheduling
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-diffvp-$$"
rm -rf "$LOCK_FILE"

source ./exp/sh/local_scheduler.sh

for i in $(seq 0 $((NUM_GPUS - 1))); do
    release_gpu "$i"
done

run_cmd="run_command_with_params_on_gpu"

# ============================================================
# Directories
# ============================================================
covariances_dir="$base_dir/covariances"
steering_vectors_dir="$base_dir/steering_vectors"
results_dir="$base_dir/evaluation"
mkdir -p "$covariances_dir" "$steering_vectors_dir" "$results_dir"

# Abstract concepts: violence and peace
topics="violence peace"
template_path="exp/datasets/eval/concepts/template_violence_peace.json"
additional_steering_params="--model_name $model_name --control_mode $control_mode --num_images_per_prompt $num_images_per_prompt --seed $seed --template_path $template_path --file_format JPEG"

# ============================================================
# STEPS 1-2: Covariances + Steering vectors
# ============================================================
echo "=============================="
echo "STEPS 1-2: Covariances + Steering vectors"
echo "Model: $model_name (estimation: $estimate_model_name)"
echo "Concepts: $topics"
echo "Started at: $(date)"
echo "=============================="

$run_cmd $python scripts/diffusion/estimate_covariances.py \
    --model_name "$estimate_model_name" \
    --control_mode "$control_mode" \
    --aggregation_mode "$aggregation_mode" \
    --num_samples "$num_covariances" \
    --output_dir "$covariances_dir" &

additional_steering_params="$additional_steering_params --covariances_dir $covariances_dir"

$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$estimate_model_name" \
    --control_mode "$control_mode" \
    --topics $topics \
    --dataset_type relaion \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir "$steering_vectors_dir" &

wait

echo "=============================="
echo "Steps 1-2 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3: Concept Translation (violence → peace)
# ============================================================
echo "=============================="
echo "STEP 3: Concept Translation"
echo "Started at: $(date)"
echo "=============================="

declare -A concept_pairs=(
    ["violence"]="peace"
)

# Concepts to steer: source, target, and unrelated at varying semantic distances
declare -a concepts_to_steer_pairs=(
    "violence:violence"
    "violence:peace"
    "violence:war"
    "violence:calm"
    "violence:anger"
    "violence:nature"
)

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"

    sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
    results_subdir="$results_dir/concept_translation/${source_concept}_to_${target_concept}__${sanitized_concept}"
    mkdir -p "$results_subdir"

    # Baseline
    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept_to_steer" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    # CASteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" &
    done

    # LEACE
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" &
    done

    # MidSteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_steer" \
            --output_dir "$results_subdir/midsteer-$strength" \
            --steering_method midsteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" &
    done

    wait
done

echo "=============================="
echo "Step 3 (concept translation) COMPLETE at: $(date)"
echo "=============================="

echo "============================================="
echo "ALL STEPS COMPLETE at: $(date)"
echo "Model: $model_name | Concepts: violence → peace"
echo "Results in: $base_dir"
echo "============================================="
