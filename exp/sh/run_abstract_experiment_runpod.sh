#!/usr/bin/env bash
# Self-contained RunPod script for abstract concept experiment (toxicity → helpfulness)
# Usage: bash exp/sh/run_abstract_experiment_runpod.sh
# Expects: HF_TOKEN env var set for gated model access

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
strengths="1.0 2.0 3.0 4.0 5.0"
python=python

model_dir_name=$(echo "$model_name" | sed 's/\//-/g')
JOB_NAME="abstract_concepts"
base_dir="/root/midsteer/exp/results/$model_dir_name/$JOB_NAME"

# ============================================================
# GPU scheduling setup (local_scheduler.sh)
# ============================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-abstract-$$"
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

# Abstract concepts
topics="toxicity helpfulness"
template_path="exp/datasets/eval/concepts/template_toxicity_helpfulness.json"
template_path_unrelated="exp/datasets/eval/concepts/template_abstract.json"

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
    --num_samples 100 \
    --output_dir "$steering_vectors_dir" &

wait

echo "=============================="
echo "Steps 1-2 COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3: Steered generation — concept evaluation
# ============================================================
echo "=============================="
echo "STEP 3: Steered generation (concept eval)"
echo "Started at: $(date)"
echo "=============================="

consistency_num_samples=1000
consistency_max_new_tokens=100
consistency_samples_per_question=1

concept_max_new_tokens=100
concept_samples_per_question=10

# Concept pair: toxicity -> helpfulness
declare -A concept_pairs=(
    ["toxicity"]="helpfulness"
)

# Concepts to evaluate: source, target, and unrelated at varying semantic distances
declare -a concepts_to_steer_pairs=(
    "toxicity:toxicity"
    "toxicity:helpfulness"
    "toxicity:sarcasm"
    "toxicity:politeness"
    "toxicity:creativity"
    "toxicity:mathematics"
)

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"

    sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
    results_subdir="$results_dir/${source_concept}_to_${target_concept}__${sanitized_concept}"
    mkdir -p "$results_subdir"

    # Use behavioral template for source/target, generic for unrelated
    if [ "$concept_to_steer" = "$source_concept" ] || [ "$concept_to_steer" = "$target_concept" ]; then
        current_template=$template_path
    else
        current_template=$template_path_unrelated
    fi

    declare -a concept_params=(--dataset_type template --template_path "$current_template" --samples_per_question "$concept_samples_per_question" --max_new_tokens "$concept_max_new_tokens" --output_dir "$results_subdir/eval")

    # Baseline (strength 0)
    $run_cmd $python scripts/llm/run_with_steering.py \
        --model_name "$model_name" \
        --layer_type "$layer_type" \
        --source_concept "$concept_to_steer" \
        --strength 0.0 \
        $additional_steering_params \
        "${concept_params[@]}" &

    # CASteer
    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name "$model_name" \
            --layer_type "$layer_type" \
            --source_concept "$concept_to_steer" \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
            --steer_type casteer \
            --strength "$strength" \
            --mu_neutral "$covariances_dir/means.pt" \
            --cov_neutral "$covariances_dir/covariances.pt" \
            $additional_steering_params \
            "${concept_params[@]}" &
    done

    # LEACE
    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name "$model_name" \
            --layer_type "$layer_type" \
            --source_concept "$concept_to_steer" \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
            --steer_type leace \
            --strength "$strength" \
            --mu_neutral "$covariances_dir/means.pt" \
            --cov_neutral "$covariances_dir/covariances.pt" \
            $additional_steering_params \
            "${concept_params[@]}" &
    done

    # MidSteer
    for strength in $strengths; do
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name "$model_name" \
            --layer_type "$layer_type" \
            --source_concept "$concept_to_steer" \
            --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
            --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
            --steer_type midsteer \
            --strength "$strength" \
            --mu_neutral "$covariances_dir/means.pt" \
            --cov_neutral "$covariances_dir/covariances.pt" \
            $additional_steering_params \
            "${concept_params[@]}" &
    done
done

wait

echo "=============================="
echo "Step 3 (concept eval) COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 3b: Concept scoring
# ============================================================
echo "=============================="
echo "STEP 3b: Concept scoring"
echo "Started at: $(date)"
echo "=============================="

for pair in "${concepts_to_steer_pairs[@]}"; do
    IFS=':' read -r source_concept concept_to_steer <<< "$pair"
    target_concept="${concept_pairs[$source_concept]}"

    sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
    results_subdir="$results_dir/${source_concept}_to_${target_concept}__${sanitized_concept}"

    $run_cmd $python scripts/llm/concept_scoring.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir/eval" &

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/eval" &
done

wait

echo "=============================="
echo "Step 3b COMPLETE at: $(date)"
echo "=============================="

# ============================================================
# STEP 4: Consistency evaluation (Alpaca + MMLU)
# ============================================================
echo "=============================="
echo "STEP 4: Consistency evaluation (Alpaca + MMLU)"
echo "Started at: $(date)"
echo "=============================="

for source_concept in "${!concept_pairs[@]}"; do
    target_concept="${concept_pairs[$source_concept]}"

    results_subdir="$results_dir/${source_concept}_to_${target_concept}"
    mkdir -p "$results_subdir"

    declare -a eval_params=(
        "--dataset_type alpaca --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/alpaca"
        "--dataset_type mmlu --num_samples $consistency_num_samples --samples_per_question $consistency_samples_per_question --max_new_tokens $consistency_max_new_tokens --output_dir $results_subdir/mmlu"
    )

    for params in "${eval_params[@]}"; do
        # Baseline
        $run_cmd $python scripts/llm/run_with_steering.py \
            --model_name "$model_name" \
            --layer_type "$layer_type" \
            --source_concept "$source_concept" \
            --strength 0.0 \
            $additional_steering_params \
            $params &

        # CASteer
        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name "$model_name" \
                --layer_type "$layer_type" \
                --source_concept "$source_concept" \
                --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
                --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
                --steer_type casteer \
                --strength "$strength" \
                --mu_neutral "$covariances_dir/means.pt" \
                --cov_neutral "$covariances_dir/covariances.pt" \
                $additional_steering_params \
                $params &
        done

        # LEACE
        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name "$model_name" \
                --layer_type "$layer_type" \
                --source_concept "$source_concept" \
                --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
                --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
                --steer_type leace \
                --strength "$strength" \
                --mu_neutral "$covariances_dir/means.pt" \
                --cov_neutral "$covariances_dir/covariances.pt" \
                $additional_steering_params \
                $params &
        done

        # MidSteer
        for strength in $strengths; do
            $run_cmd $python scripts/llm/run_with_steering.py \
                --model_name "$model_name" \
                --layer_type "$layer_type" \
                --source_concept "$source_concept" \
                --source_concept_path "$steering_vectors_dir/$source_concept.pt" \
                --target_concept_path "$steering_vectors_dir/$target_concept.pt" \
                --steer_type midsteer \
                --strength "$strength" \
                --mu_neutral "$covariances_dir/means.pt" \
                --cov_neutral "$covariances_dir/covariances.pt" \
                $additional_steering_params \
                $params &
        done
    done
done

wait

echo "=============================="
echo "Step 4 generation COMPLETE at: $(date)"
echo "=============================="

# Consistency scoring
for source_concept in "${!concept_pairs[@]}"; do
    target_concept="${concept_pairs[$source_concept]}"
    results_subdir="$results_dir/${source_concept}_to_${target_concept}"

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/alpaca" &

    $run_cmd $python scripts/llm/consistency_scoring.py \
        --dir "$results_subdir/mmlu" &
done

wait

echo "============================================="
echo "ALL STEPS COMPLETE at: $(date)"
echo "Results in: $base_dir"
echo "============================================="
