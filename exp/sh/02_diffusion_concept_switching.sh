#!/usr/bin/env bash
# 02_diffusion_concept_switching.sh
#
# Reproduces the headline diffusion concept-switching result from the paper
# (Section 5, SDXL, dogs -> cats), comparing CASteer (baseline), LEACE, and
# MidSteer across nine steering strengths. Generates 10 images per prompt for
# the source/target templates and computes CLIP-based concept scores + FID.
#
# Outputs:  $OUTPUT_DIR/{covariances,steering_vectors,evaluation}
# Estimate: ~15-20h on a single H100 sequential. Covariance estimation uses
#           sdxl-turbo for speed; generation uses full sdxl.
# Requires: HF_TOKEN. SDXL itself is not gated, but other models referenced in
#           the paper (FLUX, SANA) are.

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-sdxl}"          # sdxl | sana | sd14 | flux ...
ESTIMATE_MODEL="${ESTIMATE_MODEL:-sdxl-turbo}"
CONTROL_MODE="${CONTROL_MODE:-attn_output}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-10}"
SEED="${SEED:-42}"
STRENGTHS="${STRENGTHS:-1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/diffusion_concept_switching}"

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
EVAL_DIR="$OUTPUT_DIR/evaluation/dogs_to_cats"
mkdir -p "$EVAL_DIR"

# ----- Step 1: covariance over re-LAION captions -----------------------------
python scripts/diffusion/estimate_covariances.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --aggregation_mode all \
    --num_samples "$NUM_COV_SAMPLES" \
    --output_dir "$COV_DIR"

# ----- Step 2: per-concept steering vectors ----------------------------------
python scripts/diffusion/estimate_steering_vectors.py \
    --model_name "$ESTIMATE_MODEL" \
    --control_mode "$CONTROL_MODE" \
    --topics dog cat \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3: generate baseline + steered images ---------------------------
common_args=(
    --model_name "$MODEL"
    --control_mode "$CONTROL_MODE"
    --covariances_dir "$COV_DIR"
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
    --seed "$SEED"
)

# Baseline (no steering): generate with the source concept prompts
python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
    --generate_concept dog \
    --output_dir "$EVAL_DIR/orig"

for method in casteer leace midsteer; do
    for strength in $STRENGTHS; do
        python scripts/diffusion/run_with_steering.py "${common_args[@]}" \
            --generate_concept dog \
            --output_dir "$EVAL_DIR/${method}-${strength}" \
            --steering_method "$method" \
            --steering_strength "$strength" \
            translate \
            --source_concept_path "$SV_DIR/dog.pt" \
            --target_concept_path "$SV_DIR/cat.pt"
    done
done

# ----- Step 4: CLIP concept scores + FID -------------------------------------
python scripts/diffusion/produce_scores.py \
    --concept dog cat \
    --dir "$EVAL_DIR" \
    --batch_size 32

echo "Done. See CLIP scores in $EVAL_DIR/*/clip_scores.tsv."
echo "Plot the Pareto frontier in notebooks/produce_charts.ipynb."
