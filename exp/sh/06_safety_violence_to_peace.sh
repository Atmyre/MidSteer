#!/usr/bin/env bash
# 06_safety_violence_to_peace.sh
#
# Rebuttal experiment (revised paper, abstract/safety diffusion concepts):
# switches "violence" -> "peace" on SDXL or SANA and measures CLIP-based
# violence accuracy on (a) violence prompts (should drop to 0) and
# (b) peace prompts + 4 unrelated concepts (sadness/calm/anger/nature)
# (should NOT spuriously rise -- this is the rebuttal's key finding).
#
# Usage:    bash exp/sh/06_safety_violence_to_peace.sh [sdxl|sana]
# Outputs:  $OUTPUT_DIR/{covariances,steering_vectors,evaluation/...}

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${1:-${MODEL:-sdxl}}"
CONTROL_MODE="${CONTROL_MODE:-attn_output}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-10}"
SEED="${SEED:-42}"
STRENGTHS="${STRENGTHS:-1.0 2.0 3.0 4.0 5.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/safety_violence_to_peace_${MODEL}}"
TEMPLATE="exp/datasets/eval/concepts/template_violence_peace.json"

# SDXL/SANA use turbo/sprint variants for covariance/vector estimation
case "$MODEL" in
    sdxl)  ESTIMATE_MODEL="sdxl-turbo" ;;
    sana)  ESTIMATE_MODEL="sana-sprint" ;;
    *)     ESTIMATE_MODEL="$MODEL" ;;
esac

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
EVAL_DIR="$OUTPUT_DIR/evaluation"
mkdir -p "$EVAL_DIR"

# ----- Steps 1-2: covariance + steering vectors ------------------------------
python scripts/diffusion/estimate_covariances.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --aggregation_mode all \
    --num_samples "$NUM_COV_SAMPLES" \
    --output_dir "$COV_DIR"

python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --topics violence peace \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3: generate for each test concept --------------------------------
common_args=(
    --model_name "$MODEL"
    --control_mode "$CONTROL_MODE"
    --covariances_dir "$COV_DIR"
    --template_path "$TEMPLATE"
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
    --seed "$SEED"
    --file_format JPEG
)

# Evaluate steering on the source ("violence"), target ("peace"), and four
# unrelated concepts at varying semantic distance.
for concept in violence peace sadness calm anger nature; do
    cdir="$EVAL_DIR/violence_to_peace__${concept}"
    mkdir -p "$cdir"

    # Baseline (no steering)
    python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
        --generate_concept "$concept" \
        --output_dir "$cdir/orig"

    for method in casteer leace midsteer; do
        for strength in $STRENGTHS; do
            python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
                --generate_concept "$concept" \
                --output_dir "$cdir/${method}-${strength}" \
                --steering_method "$method" \
                --steering_strength "$strength" \
                translate \
                --source_concept_path "$SV_DIR/violence.pt" \
                --target_concept_path "$SV_DIR/peace.pt"
        done
    done

    # CLIP-based violence accuracy per directory
    python scripts/diffusion/produce_scores.py \
        --concept violence \
        --dir "$cdir" \
        --batch_size 32
done

echo "Done. CLIP violence scores per concept in $EVAL_DIR/violence_to_peace__*/."
