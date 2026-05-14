#!/usr/bin/env bash
# Rebuttal experiment: cross-validate LLM judge with GPT-4o-mini.
#
# Reproduces the main concept-switching evaluation with 1 seed/prompt (instead
# of 10), then scores outputs with both the original Llama-3.1-8B-Instruct
# judge AND GPT-4o-mini.
#
# Usage:
#   # Step 0 (once, needs OPENAI_API_KEY): generate chihuahua/muffin training Qs
#   python scripts/llm/generate_training_questions.py --topics chihuahuas muffins
#
#   # Full pipeline on GPU:
#   bash exp/sh/run_rebuttal_llm_judge.sh
#
#   # Or GPT-4o-mini scoring only (no GPU needed, after generation is done):
#   bash exp/sh/run_rebuttal_llm_judge.sh --score-gpt4o-only

set -eoux pipefail

export PYTHONPATH=.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
models=("meta-llama/Llama-2-7b-chat-hf" "Qwen/Qwen2.5-7B-Instruct")
layer_type=self_attn
num_covariances=50000
max_new_tokens=100
strengths="1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"
samples_per_question=1       # 1 seed instead of 10
scoring_model="gpt-4o-mini"  # OpenAI judge

SCORE_GPT4O_ONLY=false
if [[ "${1:-}" == "--score-gpt4o-only" ]]; then
    SCORE_GPT4O_ONLY=true
fi

# Concept pairs: source → target
declare -A concept_pairs=(
    ["horses"]="motorcycles"
    ["dogs"]="cats"
    ["chihuahuas"]="muffins"
)

# Which template concepts to evaluate per steering pair.
# Format: "steering_source:template_concept"
# - source:source → gives source CS + target CS
# - source:unrelated → gives unrelated CS (preservation)
declare -a concepts_to_steer_pairs=(
    "horses:horses"
    "horses:legislators"
    "dogs:dogs"
    "dogs:legislators"
    "chihuahuas:chihuahuas"
    "chihuahuas:legislators"
)

# All topics that need steering vectors
all_topics="horses motorcycles cats dogs chihuahuas muffins"

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
generate_for_concept() {
    local model_name=$1
    local concept_to_steer=$2
    local source_concept=$3
    local target_concept=$4
    local steering_vectors_dir=$5
    local covariances_dir=$6
    local results_subdir=$7

    local common_args=(
        --model_name "$model_name"
        --layer_type $layer_type
        --source_concept "$concept_to_steer"
        --dataset_type template
        --samples_per_question $samples_per_question
        --max_new_tokens $max_new_tokens
        --output_dir "$results_subdir/eval"
    )

    # Baseline (no steering)
    python scripts/llm/run_with_steering.py \
        "${common_args[@]}" \
        --strength 0.0 \
        --intermediate_clipping

    # Steered outputs
    for method in casteer leace midsteer; do
        for strength in $strengths; do
            python scripts/llm/run_with_steering.py \
                "${common_args[@]}" \
                --source_concept_path "$steering_vectors_dir/${source_concept}.pt" \
                --target_concept_path "$steering_vectors_dir/${target_concept}.pt" \
                --steer_type $method \
                --strength "$strength" \
                --mu_neutral "$covariances_dir/means.pt" \
                --cov_neutral "$covariances_dir/covariances.pt" \
                --intermediate_clipping
        done
    done
}

score_with_llama() {
    local results_subdir=$1
    local source_concept=$2
    local target_concept=$3
    local concept_to_steer=$4

    python scripts/llm/concept_scoring.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir/eval"
}

score_with_gpt4o() {
    local results_subdir=$1
    local source_concept=$2
    local target_concept=$3
    local concept_to_steer=$4

    python scripts/llm/concept_scoring_gpt4o.py \
        --concept "$source_concept" "$target_concept" "$concept_to_steer" \
        --dir "$results_subdir/eval" \
        --model "$scoring_model"
}

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
for model_name in "${models[@]}"; do
    model_dir_name=$(echo "$model_name" | sed 's/\//-/g')
    base_dir="exp/results/$model_dir_name/rebuttal_judge"

    if [[ "$SCORE_GPT4O_ONLY" == false ]]; then
        # ---- Step 1: Estimate covariances ----
        python scripts/llm/estimate_covariances.py \
            --model_name "$model_name" \
            --layer_type $layer_type \
            --token_aggregation_mode last \
            --num_samples $num_covariances \
            --max_new_tokens $max_new_tokens \
            --output_dir "$base_dir/covariances"

        # ---- Step 2: Generate steering vectors ----
        python scripts/llm/generate_steering_vectors.py \
            --model_name "$model_name" \
            --layer_type $layer_type \
            --topics $all_topics \
            --token_aggregation_mode last \
            --max_new_tokens 1 \
            --num_samples 1000 \
            --output_dir "$base_dir/steering_vectors"

        # ---- Step 3: Generate outputs ----
        for pair in "${concepts_to_steer_pairs[@]}"; do
            IFS=':' read -r source_concept concept_to_steer <<< "$pair"
            target_concept="${concept_pairs[$source_concept]}"

            sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
            results_subdir="$base_dir/evaluation/${source_concept}_to_${target_concept}__${sanitized_concept}"
            mkdir -p "$results_subdir"

            generate_for_concept \
                "$model_name" "$concept_to_steer" "$source_concept" "$target_concept" \
                "$base_dir/steering_vectors" "$base_dir/covariances" "$results_subdir"
        done

        # ---- Step 4: Score with Llama judge (GPU) ----
        for pair in "${concepts_to_steer_pairs[@]}"; do
            IFS=':' read -r source_concept concept_to_steer <<< "$pair"
            target_concept="${concept_pairs[$source_concept]}"
            sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
            results_subdir="$base_dir/evaluation/${source_concept}_to_${target_concept}__${sanitized_concept}"

            score_with_llama "$results_subdir" "$source_concept" "$target_concept" "$concept_to_steer"
        done
    fi

    # ---- Step 5: Score with GPT-4o-mini (API only, no GPU needed) ----
    for pair in "${concepts_to_steer_pairs[@]}"; do
        IFS=':' read -r source_concept concept_to_steer <<< "$pair"
        target_concept="${concept_pairs[$source_concept]}"
        sanitized_concept=$(echo "$concept_to_steer" | sed "s/[[:space:]'\"]/_/g")
        results_subdir="$base_dir/evaluation/${source_concept}_to_${target_concept}__${sanitized_concept}"

        score_with_gpt4o "$results_subdir" "$source_concept" "$target_concept" "$concept_to_steer"
    done
done

echo "Done. Compare concept_scores.tsv (Llama) vs concept_scores_gpt4o.tsv (GPT-4o-mini) in each eval/ directory."
