#!/usr/bin/env bash
# 05_safety_toxicity_to_helpfulness.sh
#
# Rebuttal experiment (revised paper, abstract/safety concepts): switches
# the abstract concept "toxicity" -> "helpfulness" on Llama-2-7B and measures
# Detoxify toxicity on 500 prompts from RealToxicityPrompts (toxic prompts)
# and on the "helpfulness" template prompts (to check there's no leakage of
# toxicity into safe contexts).
#
# Outputs:  $OUTPUT_DIR/{covariances,steering_vectors,evaluation/{rtp,helpfulness}}
# Requires: HF_TOKEN. Detoxify weights are auto-downloaded on first run.

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
LAYER="${LAYER:-self_attn}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
STRENGTHS="${STRENGTHS:-1.0 2.0 3.0 4.0 5.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/safety_toxicity_to_helpfulness}"

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
RTP_DIR="$OUTPUT_DIR/evaluation/rtp"
HELP_DIR="$OUTPUT_DIR/evaluation/helpfulness"
TEMPLATE="exp/datasets/eval/concepts/template_toxicity_helpfulness.json"
mkdir -p "$RTP_DIR" "$HELP_DIR"

# ----- Steps 1-2: covariance + toxicity/helpfulness steering vectors ---------
python scripts/llm/estimate_covariances.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --token_aggregation_mode last \
    --num_samples "$NUM_COV_SAMPLES" \
    --max_new_tokens 100 \
    --output_dir "$COV_DIR"

python scripts/llm/generate_steering_vectors.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --topics toxicity helpfulness \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3a: RealToxicityPrompts evaluation -------------------------------
rtp_common=(
    --model_name "$MODEL"
    --layer_type "$LAYER"
    --output_dir "$RTP_DIR"
    --num_prompts 500
    --min_toxicity 0.5
    --max_new_tokens 50
)

# Baseline (no steering)
python scripts/llm/run_rtp_eval.py "${rtp_common[@]}"

for method in casteer leace midsteer; do
    for strength in $STRENGTHS; do
        python scripts/llm/run_rtp_eval.py "${rtp_common[@]}" \
            --source_concept_path "$SV_DIR/toxicity.pt" \
            --target_concept_path "$SV_DIR/helpfulness.pt" \
            --steer_type "$method" \
            --strength "$strength" \
            --mu_neutral "$COV_DIR/means.pt" \
            --cov_neutral "$COV_DIR/covariances.pt"
    done
done

# Score the RTP outputs with Detoxify
python scripts/llm/toxicity_scoring_detoxify.py --dir "$RTP_DIR"

# ----- Step 3b: helpfulness-template prompts (check for toxicity leakage) ---
help_common=(
    --model_name "$MODEL"
    --layer_type "$LAYER"
    --source_concept helpfulness
    --dataset_type template
    --template_path "$TEMPLATE"
    --samples_per_question 1
    --max_new_tokens 100
    --output_dir "$HELP_DIR"
)

python scripts/llm/run_with_steering.py "${help_common[@]}" --strength 0.0

for method in casteer leace midsteer; do
    for strength in $STRENGTHS; do
        python scripts/llm/run_with_steering.py "${help_common[@]}" \
            --source_concept_path "$SV_DIR/toxicity.pt" \
            --target_concept_path "$SV_DIR/helpfulness.pt" \
            --steer_type "$method" \
            --strength "$strength" \
            --mu_neutral "$COV_DIR/means.pt" \
            --cov_neutral "$COV_DIR/covariances.pt"
    done
done

python scripts/llm/toxicity_scoring_detoxify.py --dir "$HELP_DIR"
python scripts/llm/helpfulness_scoring_armo.py --dir "$HELP_DIR"

echo "Done. RTP toxicity:        $RTP_DIR/toxicity_scores.tsv"
echo "      Helpfulness toxicity: $HELP_DIR/toxicity_scores.tsv (should stay near 0)"
echo "      Helpfulness ArmoRM:   $HELP_DIR/helpfulness_scores.tsv"
