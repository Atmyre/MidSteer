#!/usr/bin/env bash
# Self-contained RunPod script for safe/unsafe steering experiment (PKU-SafeRLHF)
#
# Training data: PKU-SafeRLHF contrastive pairs (same prompt, one safe + one unsafe response)
# Evaluation: BeaverTails-Evaluation 700 red-team prompts across 14 harm categories
# Scoring: beaver-dam-7b QA-moderation + detoxify toxicity classifier
#
# Usage: bash exp/sh/run_safety_experiment_runpod.sh [strengths]
# Example: bash exp/sh/run_safety_experiment_runpod.sh "1.0 2.0 3.0 4.0 5.0"

set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

# ============================================================
# Configuration
# ============================================================
model_name="meta-llama/Llama-2-7b-chat-hf"
layer_type="self_attn"
num_covariances=5000
token_aggregation_mode="last"
max_new_tokens=100
strengths="${1:-1.0 2.0 3.0 4.0 5.0}"
python=python

model_dir_name=$(echo "$model_name" | sed 's/\//-/g')
JOB_NAME="safety_pku"
base_dir="/root/midsteer/exp/results/$model_dir_name/$JOB_NAME"

# ============================================================
# GPU scheduling setup
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-safety-$$"
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

# PKU-SafeRLHF concepts: unsafe → safe
topics="unsafe safe"

# BeaverTails-Evaluation prompts (700 red-team prompts, 14 harm categories)
# These are direct prompts (no {} template), but TemplateDataset handles this fine
beavertails_template="exp/datasets/eval/concepts/beavertails_eval.json"

additional_steering_params=""

# ============================================================
# STEP 1 & 2: Covariances + Steering vectors (in parallel)
# ============================================================
echo "=============================="
echo "STEPS 1-2: Covariances + Steering vectors"
echo "Started at: $(date)"
echo "=============================="

$run_cmd $python scripts/llm/estimate_covariances.py \
    --model_name "$model_name" \
    --layer_type "$layer_type" \
    --token_aggregation_mode "$token_aggregation_mode" \
    --num_samples "$num_covariances" \
    --max_new_tokens "$max_new_tokens" \
    --output_dir "$covariances_dir" &

$run_cmd $python scripts/llm/generate_steering_vectors.py \
    --model_name "$model_name" \
    --layer_type "$layer_type" \
    --topics $topics \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir "$steering_vectors_dir" &

wait

echo "=============================="
echo "Steps 1-2 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3: Steered generation on BeaverTails-Evaluation
# ============================================================
echo "=============================="
echo "STEP 3: Steered generation (BeaverTails 700 prompts)"
echo "Started at: $(date)"
echo "=============================="

# For BeaverTails prompts, source_concept is just a label (no template substitution needed)
# since the prompts don't contain {}
concept_max_new_tokens=100
concept_samples_per_question=1

SOURCE=unsafe
TARGET=safe

beavertails_subdir="$results_dir/${SOURCE}_to_${TARGET}__beavertails"
mkdir -p "$beavertails_subdir"

declare -a bt_params=(--dataset_type template --template_path "$beavertails_template" --samples_per_question "$concept_samples_per_question" --max_new_tokens "$concept_max_new_tokens" --output_dir "$beavertails_subdir/eval")

# Baseline (no steering)
$run_cmd $python scripts/llm/run_with_steering.py \
    --model_name "$model_name" \
    --layer_type "$layer_type" \
    --source_concept "safety_eval" \
    --strength 0.0 \
    $additional_steering_params \
    "${bt_params[@]}" &

# CASteer
for strength in $strengths; do
    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name "$model_name" \
        --layer_type "$layer_type" \
        --source_concept "safety_eval" \
        --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
        --target_concept_path "$steering_vectors_dir/$TARGET.pt" \
        --steer_type casteer \
        --strength "$strength" \
        --mu_neutral "$covariances_dir/means.pt" \
        --cov_neutral "$covariances_dir/covariances.pt" \
        $additional_steering_params \
        "${bt_params[@]}" &
done

# LEACE
for strength in $strengths; do
    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name "$model_name" \
        --layer_type "$layer_type" \
        --source_concept "safety_eval" \
        --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
        --target_concept_path "$steering_vectors_dir/$TARGET.pt" \
        --steer_type leace \
        --strength "$strength" \
        --mu_neutral "$covariances_dir/means.pt" \
        --cov_neutral "$covariances_dir/covariances.pt" \
        $additional_steering_params \
        "${bt_params[@]}" &
done

# MidSteer
for strength in $strengths; do
    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name "$model_name" \
        --layer_type "$layer_type" \
        --source_concept "safety_eval" \
        --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
        --target_concept_path "$steering_vectors_dir/$TARGET.pt" \
        --steer_type midsteer \
        --strength "$strength" \
        --mu_neutral "$covariances_dir/means.pt" \
        --cov_neutral "$covariances_dir/covariances.pt" \
        $additional_steering_params \
        "${bt_params[@]}" &
done

wait

echo "=============================="
echo "Step 3 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 4: Scoring with beaver-dam-7b + detoxify
# ============================================================
echo "=============================="
echo "STEP 4: Safety scoring (beaver-dam-7b + detoxify)"
echo "Started at: $(date)"
echo "=============================="

# beaver-dam-7b QA-moderation (context-aware safety)
$run_cmd $python scripts/llm/safety_scoring_beaverdam.py \
    --dir "$beavertails_subdir/eval" &

# detoxify (text-level toxicity, for cross-validation)
$run_cmd $python scripts/llm/toxicity_scoring_detoxify.py \
    --dir "$beavertails_subdir/eval" &

wait

echo "=============================="
echo "Step 4 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 5: Consistency evaluation (Alpaca + MMLU)
# ============================================================
echo "=============================="
echo "STEP 5: Consistency evaluation (Alpaca + MMLU)"
echo "Started at: $(date)"
echo "=============================="

consistency_num_samples=1000
consistency_max_new_tokens=100
consistency_samples_per_question=1

consistency_subdir="$results_dir/${SOURCE}_to_${TARGET}"
mkdir -p "$consistency_subdir"

declare -a eval_params=(
    "--dataset_type alpaca --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $consistency_subdir/alpaca"
    "--dataset_type mmlu --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $consistency_subdir/mmlu"
)

for params in "${eval_params[@]}"; do
    # Baseline
    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name "$model_name" \
        --layer_type "$layer_type" \
        --source_concept "$SOURCE" \
        --strength 0.0 \
        $additional_steering_params \
        $params &

    for strength in $strengths; do
        for method in casteer leace midsteer; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name "$model_name" \
                --layer_type "$layer_type" \
                --source_concept "$SOURCE" \
                --source_concept_path "$steering_vectors_dir/$SOURCE.pt" \
                --target_concept_path "$steering_vectors_dir/$TARGET.pt" \
                --steer_type "$method" \
                --strength "$strength" \
                --mu_neutral "$covariances_dir/means.pt" \
                --cov_neutral "$covariances_dir/covariances.pt" \
                $additional_steering_params \
                $params &
        done
    done
done

wait

# Consistency scoring
for subdir in "$consistency_subdir/alpaca" "$consistency_subdir/mmlu"; do
    $run_cmd $python scripts/llm/consistency_scoring.py --dir "$subdir" &
done

wait

echo "=============================="
echo "Step 5 COMPLETE at: $(date)"
echo "=============================="

echo "=============================="
echo "ALL STEPS COMPLETE at: $(date)"
echo "Results in: $base_dir"
echo "=============================="
