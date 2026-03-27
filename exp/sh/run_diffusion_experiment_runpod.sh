#!/usr/bin/env bash
# Self-contained RunPod script for diffusion steering experiments (SDXL / SANA)
# Usage: bash exp/sh/run_diffusion_experiment_runpod.sh <model_name> [strengths]
#   model_name: sdxl or sana
#   strengths: quoted space-separated list (default: "1.0 2.0 3.0 4.0 5.0")
#
# Examples:
#   bash exp/sh/run_diffusion_experiment_runpod.sh sdxl
#   bash exp/sh/run_diffusion_experiment_runpod.sh sana "1.0 2.0 3.0"

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

# ============================================================
# Install correct diffusers version
# ============================================================
if [[ "$model_name" == "sana"* ]]; then
    # SANA needs newer diffusers for SanaPipeline/SanaSprintPipeline
    if ! python -c "from diffusers import SanaSprintPipeline" 2>/dev/null; then
        echo "Installing diffusers with SANA support..."
        pip install -q 'diffusers>=0.32.0'
    fi
elif [[ "$model_name" == "sdxl"* ]]; then
    # SDXL works with 0.30.0, but let's make sure diffusers is installed
    if ! python -c "from diffusers import DiffusionPipeline" 2>/dev/null; then
        pip install -q 'diffusers==0.30.0'
    fi
fi

# ============================================================
# Model mapping for estimation
# ============================================================
estimate_model_name=$model_name
if [[ $model_name == "sdxl" ]]; then
    estimate_model_name="sdxl-turbo"
elif [[ $model_name == "sana" ]]; then
    estimate_model_name="sana-sprint"
fi

JOB_NAME="diffusion_${model_name}"
base_dir="/root/midsteer/exp/results/$model_name/$JOB_NAME"

# ============================================================
# GPU scheduling setup
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-diffusion-$$"
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

topics="horse motorcycle snoopy mickey chihuahua muffin"
additional_steering_params="--model_name $model_name --control_mode $control_mode --num_images_per_prompt $num_images_per_prompt --seed $seed"

# ============================================================
# STEP 1 & 2: Covariances + Steering vectors (in parallel)
# ============================================================
echo "=============================="
echo "STEPS 1-2: Covariances + Steering vectors"
echo "Model: $model_name (estimation model: $estimate_model_name)"
echo "Started at: $(date)"
echo "=============================="

if [ $num_covariances -gt 0 ]; then
    $run_cmd $python scripts/diffusion/estimate_covariances.py \
        --model_name "$estimate_model_name" \
        --control_mode "$control_mode" \
        --aggregation_mode "$aggregation_mode" \
        --num_samples "$num_covariances" \
        --output_dir "$covariances_dir" &

    additional_steering_params="$additional_steering_params --covariances_dir $covariances_dir"
fi

$run_cmd $python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$estimate_model_name" \
    --control_mode "$control_mode" \
    --topics $topics \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir "$steering_vectors_dir" &

wait

echo "=============================="
echo "Steps 1-2 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3: Concept Translation
# ============================================================
echo "=============================="
echo "STEP 3: Concept Translation"
echo "Started at: $(date)"
echo "=============================="

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

    sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
    results_subdir="$results_dir/concept_translation/${source_concept}_to_${target_concept}__${sanitized_concept}"
    mkdir -p "$results_subdir"

    # Baseline (no steering)
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
# STEP 4: Concept Erasure
# ============================================================
echo "=============================="
echo "STEP 4: Concept Erasure"
echo "Started at: $(date)"
echo "=============================="

declare -a concepts_to_remove_pairs=(
    "snoopy:snoopy"
    "snoopy:mickey"
    "snoopy:spongebob"
    "snoopy:pikachu"
    "snoopy:dog"
    "snoopy:legislator"
    "horse:horse"
    "horse:motorcycle"
    "horse:cow"
    "horse:pig"
    "horse:dog"
    "horse:legislator"
    "chihuahua:chihuahua"
    "chihuahua:muffin"
    "chihuahua:wolf"
    "chihuahua:cat"
    "chihuahua:dog"
    "chihuahua:legislator"
)

for pair in "${concepts_to_remove_pairs[@]}"; do
    IFS=':' read -r concept_to_remove concept_to_generate <<< "$pair"

    results_subdir="$results_dir/concept_erasure/${concept_to_remove}_${concept_to_generate}"
    mkdir -p "$results_subdir"

    # Baseline
    $run_cmd $python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept_to_generate" \
        --output_dir "$results_subdir/orig" \
        $additional_steering_params &

    # CASteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/casteer-$strength" \
            --steering_method casteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            erase \
            --concept_path "$steering_vectors_dir/$concept_to_remove.pt" &
    done

    # LEACE
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/leace-$strength" \
            --steering_method leace \
            --steering_strength "$strength" \
            $additional_steering_params \
            erase \
            --concept_path "$steering_vectors_dir/$concept_to_remove.pt" &
    done

    # MidSteer
    for strength in $strengths; do
        $run_cmd $python scripts/diffusion/run_with_steering.py \
            --generate_concept "$concept_to_generate" \
            --output_dir "$results_subdir/midsteer-$strength" \
            --steering_method midsteer \
            --steering_strength "$strength" \
            $additional_steering_params \
            erase \
            --concept_path "$steering_vectors_dir/$concept_to_remove.pt" &
    done

    wait
done

echo "============================================="
echo "ALL STEPS COMPLETE at: $(date)"
echo "Model: $model_name"
echo "Results in: $base_dir"
echo "============================================="
