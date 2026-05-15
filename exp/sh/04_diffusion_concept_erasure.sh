#!/usr/bin/env bash
# 04_diffusion_concept_erasure.sh
#
# Concept-erasure variant of the diffusion experiment (paper Appendix J).
# Uses the 'erase' subcommand of scripts/diffusion/run_with_steering.py, which
# takes a single --concept_path (no target). LEACE and MidSteer coincide in
# this mode; CASteer is the baseline.
#
# Erases "dog" from SDXL and evaluates CLIP score + FID on dog-templated
# prompts.

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-sdxl}"
ESTIMATE_MODEL="${ESTIMATE_MODEL:-sdxl-turbo}"
CONTROL_MODE="${CONTROL_MODE:-attn_output}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-10}"
SEED="${SEED:-42}"
STRENGTHS="${STRENGTHS:-0.5 1.0 1.5 2.0 2.5 3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/diffusion_concept_erasure}"

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
EVAL_DIR="$OUTPUT_DIR/evaluation/erase_dog"
mkdir -p "$EVAL_DIR"

# ----- Steps 1-2: covariance + steering vector -------------------------------
python scripts/diffusion/estimate_covariances.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --aggregation_mode all \
    --num_samples "$NUM_COV_SAMPLES" \
    --output_dir "$COV_DIR"

python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --topics dog \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3: generate baseline + erasure-steered images --------------------
common_args=(
    --model_name "$MODEL"
    --control_mode "$CONTROL_MODE"
    --covariances_dir "$COV_DIR"
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
    --seed "$SEED"
)

python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
    --generate_concept dog \
    --output_dir "$EVAL_DIR/orig"

for method in casteer leace; do  # midsteer == leace in erasure mode
    for strength in $STRENGTHS; do
        python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
            --generate_concept dog \
            --output_dir "$EVAL_DIR/${method}-${strength}" \
            --steering_method "$method" \
            --steering_strength "$strength" \
            erase \
            --concept_path "$SV_DIR/dog.pt"
    done
done

# ----- Step 4: score ----------------------------------------------------------
python scripts/diffusion/produce_scores.py \
    --concept dog \
    --dir "$EVAL_DIR" \
    --batch_size 32

echo "Done. Erasure CLIP scores in $EVAL_DIR/."
