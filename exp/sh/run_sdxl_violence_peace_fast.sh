#!/usr/bin/env bash
# RunPod script for SDXL violence→peace experiment (fast mode: 512x512, 5 imgs/prompt)
#
# Usage: bash exp/sh/run_sdxl_violence_peace_fast.sh [strengths]
# Example: bash exp/sh/run_sdxl_violence_peace_fast.sh "1.0 2.0 3.0 4.0 5.0"

set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

# ============================================================
# Configuration
# ============================================================
model_name="sdxl"
estimate_model_name="sdxl-turbo"  # Use turbo variant for faster estimation
control_mode="attn_output"
num_covariances=50000
aggregation_mode="all"
num_images_per_prompt=5           # Was 10, halved for speed
resolution=512                    # Was 1024, quartered pixel count
seed=42
strengths="${1:-1.0 2.0 3.0 4.0 5.0}"
python=python

JOB_NAME="violence_peace"
base_dir="/root/midsteer/exp/results/$model_name/$JOB_NAME"

# ============================================================
# GPU scheduling
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-sdxlvp-$$"
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

topics="violence peace"
template_path="exp/datasets/eval/concepts/template_violence_peace.json"
additional_steering_params="--model_name $model_name --control_mode $control_mode --num_images_per_prompt $num_images_per_prompt --resolution $resolution --seed $seed --template_path $template_path --file_format JPEG"

# ============================================================
# STEPS 1-2: Covariances + Steering vectors
# ============================================================
echo "=============================="
echo "STEPS 1-2: Covariances + Steering vectors"
echo "Model: $model_name (estimation: $estimate_model_name)"
echo "Resolution: ${resolution}x${resolution}, Images/prompt: $num_images_per_prompt"
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

# ============================================================
# STEP 4: CLIP + FID scoring
# ============================================================
echo "=============================="
echo "STEP 4: CLIP + FID scoring"
echo "Started at: $(date)"
echo "=============================="

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"

    sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
    results_subdir="$results_dir/concept_translation/${source_concept}_to_${target_concept}__${sanitized_concept}"

    $run_cmd $python scripts/diffusion/produce_scores.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir" \
        --num_workers 4 \
        --batch_size 32 &
done

wait

echo "=============================="
echo "Step 4 COMPLETE at: $(date)"
echo "=============================="

echo "============================================="
echo "ALL STEPS COMPLETE at: $(date)"
echo "Model: SDXL | Resolution: ${resolution}x${resolution} | Images/prompt: $num_images_per_prompt"
echo "Results in: $base_dir"
echo "============================================="
