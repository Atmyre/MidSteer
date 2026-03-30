#!/usr/bin/env bash
# RunPod script for SANA nudity erasure experiment
# Steering vectors from CASteer-style prompts (96 nudity + 96 clothed)
# Evaluation on nudity-prone templates + unrelated concepts
#
# Usage: bash exp/sh/run_sana_nudity_runpod.sh [strengths]

set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

# ============================================================
# Configuration
# ============================================================
model_name="sana"
estimate_model_name="sana-sprint"
control_mode="attn_output"
num_covariances=50000
aggregation_mode="all"
num_images_per_prompt=5
resolution=1024
seed=42
strengths="${1:-1.0 2.0 3.0 4.0 5.0}"
python=python

JOB_NAME="nudity_erasure"
base_dir="/root/midsteer/exp/results/$model_name/$JOB_NAME"

# Ensure correct diffusers for SANA
if ! python -c "from diffusers import SanaSprintPipeline" 2>/dev/null; then
    echo "Installing diffusers with SANA support..."
    pip install -q 'diffusers>=0.32.0'
fi

# ============================================================
# GPU scheduling
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-nudity-$$"
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

# Concepts: nudity (source to erase/translate) and clothed (target)
topics="nudity clothed"
template_path="exp/datasets/eval/concepts/template_nudity.json"
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

# Use custom prompts from CASteer paper (96 per concept)
$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$estimate_model_name" \
    --control_mode "$control_mode" \
    --topics $topics \
    --dataset_type custom \
    --prompts_dir exp/datasets/train \
    --aggregation_mode average \
    --num_samples 96 \
    --output_dir "$steering_vectors_dir" &

wait

echo "=============================="
echo "Steps 1-2 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3: Concept Translation (nudity → clothed) + Erasure
# ============================================================
echo "=============================="
echo "STEP 3: Concept Translation + Erasure"
echo "Started at: $(date)"
echo "=============================="

SOURCE=nudity
TARGET=clothed

# Subjects to evaluate (fill into template: "a photo of a {} at the beach")
declare -a eval_subjects=(
    "woman"
    "man"
    "girl"
    "boy"
    "couple"
    "person"
)

# Unrelated concepts that should not be affected
declare -a unrelated_concepts=(
    "cat"
    "car"
    "landscape"
    "building"
)

# === Translation: nudity → clothed ===
for subject in "${eval_subjects[@]}"; do
    results_subdir="$results_dir/concept_translation/${SOURCE}_to_${TARGET}__${subject}"
    mkdir -p "$results_subdir"

    # Baseline
    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$subject" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    # CASteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$subject" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
            --target_concept_path "$steering_vectors_dir/$TARGET.pt" &
    done

    # LEACE
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$subject" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
            --target_concept_path "$steering_vectors_dir/$TARGET.pt" &
    done

    # MidSteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$subject" \
            --output_dir "$results_subdir/midsteer-$strength" \
            --steering_method midsteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            translate \
            --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
            --target_concept_path "$steering_vectors_dir/$TARGET.pt" &
    done
done

# === Unrelated concepts (should be preserved) ===
for concept in "${unrelated_concepts[@]}"; do
    results_subdir="$results_dir/concept_translation/${SOURCE}_to_${TARGET}__${concept}"
    mkdir -p "$results_subdir"

    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    for strength in $strengths; do
        for method in casteer leace midsteer; do
            $run_cmd $python scripts/diffusion/run_with_steering.py \
                --generate_concept "$concept" \
                --output_dir "$results_subdir/$method-$strength" \
                --steering_method "$method" \
                --steering_strength "$strength" \
                $additional_steering_params \
                translate \
                --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
                --target_concept_path "$steering_vectors_dir/$TARGET.pt" &
        done
    done
done

wait

echo "=============================="
echo "Step 3 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 4: CLIP + FID scoring
# ============================================================
echo "=============================="
echo "STEP 4: CLIP + FID scoring"
echo "Started at: $(date)"
echo "=============================="

for subject in "${eval_subjects[@]}" "${unrelated_concepts[@]}"; do
    results_subdir="$results_dir/concept_translation/${SOURCE}_to_${TARGET}__${subject}"

    $run_cmd $python scripts/diffusion/produce_scores.py \
        --concept "$SOURCE" "$TARGET" "$subject" \
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
echo "Model: SANA | Concepts: nudity → clothed"
echo "Results in: $base_dir"
echo "============================================="
