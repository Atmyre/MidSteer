#!/usr/bin/env bash
# 03_llm_concept_erasure.sh
#
# Concept-erasure variant of the main LLM experiment (paper Appendix J).
# In erasure mode there is no target concept; LEACE and MidSteer coincide
# (see Section 4.3). Omit --target_concept_path to switch into erasure mode.
#
# Erases "horses" from Llama-2-7B and evaluates concept-presence + Alpaca/MMLU
# consistency. Compare against CASteer-style erasure (steer away from horses).
#
# Outputs:  $OUTPUT_DIR/{covariances,steering_vectors,evaluation}

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
LAYER="${LAYER:-self_attn}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
STRENGTHS="${STRENGTHS:-0.5 1.0 1.5 2.0 2.5 3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/llm_concept_erasure}"

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
EVAL_DIR="$OUTPUT_DIR/evaluation/erase_horses"
mkdir -p "$EVAL_DIR"

# ----- Steps 1-2: covariance + steering vector (single concept) --------------
python scripts/llm/estimate_covariances.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --token_aggregation_mode all \
    --num_samples "$NUM_COV_SAMPLES" \
    --max_new_tokens 100 \
    --output_dir "$COV_DIR"

python scripts/llm/generate_steering_vectors.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --topics horses \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3: generate with erasure mode ------------------------------------
common_args=(
    --model_name "$MODEL"
    --layer_type "$LAYER"
    --source_concept horses
    --dataset_type template
    --samples_per_question 10
    --max_new_tokens 100
    --output_dir "$EVAL_DIR/eval"
)

# Baseline (no steering)
python scripts/llm/run_with_steering.py "${common_args[@]}" --strength 0.0

# Erasure: pass only --source_concept_path, omit --target_concept_path
for method in casteer leace; do  # midsteer == leace in erasure mode
    for strength in $STRENGTHS; do
        python scripts/llm/run_with_steering.py "${common_args[@]}" \
            --source_concept_path "$SV_DIR/horses.pt" \
            --steer_type "$method" \
            --strength "$strength" \
            --mu_neutral "$COV_DIR/means.pt" \
            --cov_neutral "$COV_DIR/covariances.pt"
    done
done

# ----- Step 4: score ----------------------------------------------------------
python scripts/llm/concept_scoring.py \
    --concept horses \
    --dir "$EVAL_DIR/eval"

python scripts/llm/consistency_scoring.py \
    --dir "$EVAL_DIR/eval"

echo "Done. Erasure scores in $EVAL_DIR/eval/."
