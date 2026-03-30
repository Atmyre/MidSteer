#!/usr/bin/env bash
set -eoux pipefail

# Local runner for toxicity -> helpfulness abstract concept experiment
# Usage: bash exp/sh/run_abstract_experiment_local.sh

export PYTHONPATH=.

BASE=exp/results/llama2-7b-abstract
MODEL=meta-llama/Llama-2-7b-chat-hf
LAYER_TYPE=self_attn

TEMPLATE_BEHAVIORAL=exp/datasets/eval/concepts/template_toxicity_helpfulness.json
TEMPLATE_ABSTRACT=exp/datasets/eval/concepts/template_abstract.json

STRENGTHS="1.0 1.5 2.0 2.5 3.0"

concept_max_new_tokens=100
concept_samples_per_question=10

consistency_num_samples=1000
consistency_max_new_tokens=100
consistency_samples_per_question=1

SOURCE=toxicity
TARGET=helpfulness

# Unrelated concepts at varying semantic distances
UNRELATED="sarcasm politeness creativity mathematics"

RESULTS=$BASE/evaluation

# ============================================================
# Step 3: Steered generation
# ============================================================

run_steering() {
    local concept_to_steer=$1
    local template=$2
    local subdir=$3

    mkdir -p "$subdir"

    # Baseline (no steering)
    python3 scripts/llm/run_with_steering.py \
        --model_name $MODEL \
        --layer_type $LAYER_TYPE \
        --source_concept "$concept_to_steer" \
        --strength 0.0 \
        --dataset_type template \
        --template_path "$template" \
        --samples_per_question $concept_samples_per_question \
        --max_new_tokens $concept_max_new_tokens \
        --output_dir "$subdir"

    for strength in $STRENGTHS; do
        for method in casteer leace midsteer; do
            python3 scripts/llm/run_with_steering.py \
                --model_name $MODEL \
                --layer_type $LAYER_TYPE \
                --source_concept "$concept_to_steer" \
                --source_concept_path $BASE/steering_vectors/$SOURCE.pt \
                --target_concept_path $BASE/steering_vectors/$TARGET.pt \
                --steer_type $method \
                --strength $strength \
                --mu_neutral $BASE/covariances/means.pt \
                --cov_neutral $BASE/covariances/covariances.pt \
                --dataset_type template \
                --template_path "$template" \
                --samples_per_question $concept_samples_per_question \
                --max_new_tokens $concept_max_new_tokens \
                --output_dir "$subdir"
        done
    done
}

echo "=== Step 3a: Steering on source concept ($SOURCE) ==="
run_steering "$SOURCE" "$TEMPLATE_BEHAVIORAL" "$RESULTS/${SOURCE}_to_${TARGET}__${SOURCE}/eval"

echo "=== Step 3b: Steering on target concept ($TARGET) ==="
run_steering "$TARGET" "$TEMPLATE_BEHAVIORAL" "$RESULTS/${SOURCE}_to_${TARGET}__${TARGET}/eval"

echo "=== Step 3c: Steering on unrelated concepts ==="
for concept in $UNRELATED; do
    echo "--- Unrelated: $concept ---"
    run_steering "$concept" "$TEMPLATE_ABSTRACT" "$RESULTS/${SOURCE}_to_${TARGET}__${concept}/eval"
done

echo "=== Step 3d: Steering on MMLU and Alpaca ==="
mmlu_dir="$RESULTS/${SOURCE}_to_${TARGET}/mmlu"
alpaca_dir="$RESULTS/${SOURCE}_to_${TARGET}/alpaca"
mkdir -p "$mmlu_dir" "$alpaca_dir"

for dataset_info in "mmlu:$mmlu_dir" "alpaca:$alpaca_dir"; do
    IFS=':' read -r ds dir <<< "$dataset_info"

    python3 scripts/llm/run_with_steering.py \
        --model_name $MODEL \
        --layer_type $LAYER_TYPE \
        --source_concept $SOURCE \
        --strength 0.0 \
        --dataset_type $ds \
        --num_samples $consistency_num_samples \
        --samples_per_question $consistency_samples_per_question \
        --max_new_tokens $consistency_max_new_tokens \
        --output_dir "$dir"

    for strength in $STRENGTHS; do
        for method in casteer leace midsteer; do
            python3 scripts/llm/run_with_steering.py \
                --model_name $MODEL \
                --layer_type $LAYER_TYPE \
                --source_concept $SOURCE \
                --source_concept_path $BASE/steering_vectors/$SOURCE.pt \
                --target_concept_path $BASE/steering_vectors/$TARGET.pt \
                --steer_type $method \
                --strength $strength \
                --mu_neutral $BASE/covariances/means.pt \
                --cov_neutral $BASE/covariances/covariances.pt \
                --dataset_type $ds \
                --num_samples $consistency_num_samples \
                --samples_per_question $consistency_samples_per_question \
                --max_new_tokens $consistency_max_new_tokens \
                --output_dir "$dir"
        done
    done
done

# ============================================================
# Step 4: Scoring
# ============================================================

echo "=== Step 4a: Concept scoring ==="
for concept in $SOURCE $TARGET $UNRELATED; do
    subdir="$RESULTS/${SOURCE}_to_${TARGET}__${concept}/eval"
    python3 scripts/llm/concept_scoring.py \
        --concept "$SOURCE" "$TARGET" "$concept" \
        --dir "$subdir"
done

echo "=== Step 4b: Consistency scoring ==="
for concept in $SOURCE $TARGET $UNRELATED; do
    subdir="$RESULTS/${SOURCE}_to_${TARGET}__${concept}/eval"
    python3 scripts/llm/consistency_scoring.py \
        --dir "$subdir"
done

python3 scripts/llm/consistency_scoring.py --dir "$mmlu_dir"
python3 scripts/llm/consistency_scoring.py --dir "$alpaca_dir"

echo "=== ALL DONE ==="
