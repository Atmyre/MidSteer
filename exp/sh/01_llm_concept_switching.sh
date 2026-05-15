#!/usr/bin/env bash
# 01_llm_concept_switching.sh
#
# Reproduces the headline LLM concept-switching result from the paper
# (Section 5, Llama-2-7B, horses -> motorcycles), comparing CASteer (baseline),
# LEACE, and MidSteer across nine steering strengths.
#
# Outputs:   $OUTPUT_DIR/{covariances,steering_vectors,evaluation}
# Estimate:  ~10-14h on a single H100 (sequential).
#            Add '&' to the inner loops and rely on multiple GPUs to parallelize.
# Requires:  HF_TOKEN with access to meta-llama/Llama-2-7b-chat-hf
#            and meta-llama/Llama-3.1-8B-Instruct (the judge).

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
LAYER="${LAYER:-self_attn}"
NUM_COV_SAMPLES="${NUM_COV_SAMPLES:-50000}"
STRENGTHS="${STRENGTHS:-1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/llm_concept_switching}"

COV_DIR="$OUTPUT_DIR/covariances"
SV_DIR="$OUTPUT_DIR/steering_vectors"
EVAL_DIR="$OUTPUT_DIR/evaluation/horses_to_motorcycles"
mkdir -p "$EVAL_DIR/eval"

# ----- Step 1: neutral-prompt covariance ($\Sigma_{XX}$ in the paper) --------
python scripts/llm/estimate_covariances.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --token_aggregation_mode all \
    --num_samples "$NUM_COV_SAMPLES" \
    --max_new_tokens 100 \
    --output_dir "$COV_DIR"

# ----- Step 2: per-concept steering vectors (source + target) ----------------
python scripts/llm/generate_steering_vectors.py \
    --model_name "$MODEL" \
    --layer_type "$LAYER" \
    --topics horses motorcycles \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir "$SV_DIR"

# ----- Step 3: generate baseline + steered outputs ---------------------------
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

for method in casteer leace midsteer; do
    for strength in $STRENGTHS; do
        python scripts/llm/run_with_steering.py "${common_args[@]}" \
            --source_concept_path "$SV_DIR/horses.pt" \
            --target_concept_path "$SV_DIR/motorcycles.pt" \
            --steer_type "$method" \
            --strength "$strength" \
            --mu_neutral "$COV_DIR/means.pt" \
            --cov_neutral "$COV_DIR/covariances.pt"
    done
done

# ----- Step 4: score (LLM-as-judge concept presence + Alpaca/MMLU BLEU) -----
python scripts/llm/concept_scoring.py \
    --concept horses motorcycles \
    --dir "$EVAL_DIR/eval"

python scripts/llm/consistency_scoring.py \
    --dir "$EVAL_DIR/eval"

echo "Done. See $EVAL_DIR/eval/concept_scores.tsv and consistency_scores.tsv."
echo "Plot the Pareto frontier in notebooks/produce_charts.ipynb."
