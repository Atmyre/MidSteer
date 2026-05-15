#!/usr/bin/env bash
# 07_gpt4o_judge_crossval.sh
#
# Rebuttal experiment (revised paper): cross-validate the LLM-as-judge concept
# scores by running a second judge (GPT-4o-mini via OpenAI API) over the same
# generations. Confirms that the relative ranking CASteer < LEACE < MidSteer
# is judge-independent.
#
# Assumes you have already generated outputs via 01_llm_concept_switching.sh.
# Re-scores those outputs with both the original Llama-3.1-8B-Instruct judge
# and GPT-4o-mini, writing two parallel TSVs per evaluation directory.
#
# Usage:    bash exp/sh/07_gpt4o_judge_crossval.sh [path/to/evaluation/dir]
# Requires: OPENAI_API_KEY for the GPT-4o-mini judge (no GPU needed for it).

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set; see .env.example}"
export PYTHONPATH="${PYTHONPATH:-.}"

# Default to the output of 01_llm_concept_switching.sh
EVAL_PARENT="${1:-./results/llm_concept_switching/evaluation/horses_to_motorcycles}"
EVAL_DIR="$EVAL_PARENT/eval"

if [[ ! -d "$EVAL_DIR" ]]; then
    echo "Error: $EVAL_DIR not found. Run 01_llm_concept_switching.sh first."
    exit 1
fi

# Llama-3.1-8B judge (original, requires HF_TOKEN + GPU)
if [[ -n "${HF_TOKEN:-}" ]]; then
    python scripts/llm/concept_scoring.py \
        --concept horses motorcycles \
        --dir "$EVAL_DIR"
else
    echo "Skipping Llama judge (HF_TOKEN not set). Set it to enable."
fi

# GPT-4o-mini judge (rebuttal addition, API-only)
python scripts/llm/concept_scoring_gpt4o.py \
    --concept horses motorcycles \
    --dir "$EVAL_DIR" \
    --model gpt-4o-mini

echo "Done. Compare:"
echo "  Llama judge:        $EVAL_DIR/concept_scores.tsv"
echo "  GPT-4o-mini judge:  $EVAL_DIR/concept_scores_gpt4o.tsv"
