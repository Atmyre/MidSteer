#!/usr/bin/env bash
# Run RealToxicityPrompts evaluation for all steering methods
set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

model_name="meta-llama/Llama-2-7b-chat-hf"
base="/root/midsteer/exp/results/meta-llama-Llama-2-7b-chat-hf/abstract_concepts"
sv_dir="$base/steering_vectors"
cov_dir="$base/covariances"
output_dir="$base/evaluation/rtp"
strengths="1.0 2.0 3.0 4.0 5.0"

NUM_GPUS=$(nvidia-smi -L | wc -l)
export LOCK_FILE="/tmp/mmsteer-gpu-rtp-$$"
rm -rf "$LOCK_FILE"
source ./exp/sh/local_scheduler.sh
for i in $(seq 0 $((NUM_GPUS - 1))); do
    release_gpu "$i"
done
run_cmd="run_command_with_params_on_gpu"

common="--model_name $model_name --layer_type self_attn --output_dir $output_dir --num_prompts 500 --min_toxicity 0.5 --max_new_tokens 50"

echo "=== RTP Evaluation ==="
echo "Started at: $(date)"

# Baseline
$run_cmd python scripts/llm/run_rtp_eval.py $common &

# CASteer
for s in $strengths; do
    $run_cmd python scripts/llm/run_rtp_eval.py $common \
        --source_concept_path $sv_dir/toxicity.pt \
        --target_concept_path $sv_dir/helpfulness.pt \
        --steer_type casteer --strength $s \
        --mu_neutral $cov_dir/means.pt --cov_neutral $cov_dir/covariances.pt &
done

# LEACE
for s in $strengths; do
    $run_cmd python scripts/llm/run_rtp_eval.py $common \
        --source_concept_path $sv_dir/toxicity.pt \
        --target_concept_path $sv_dir/helpfulness.pt \
        --steer_type leace --strength $s \
        --mu_neutral $cov_dir/means.pt --cov_neutral $cov_dir/covariances.pt &
done

# MidSteer
for s in $strengths; do
    $run_cmd python scripts/llm/run_rtp_eval.py $common \
        --source_concept_path $sv_dir/toxicity.pt \
        --target_concept_path $sv_dir/helpfulness.pt \
        --steer_type midsteer --strength $s \
        --mu_neutral $cov_dir/means.pt --cov_neutral $cov_dir/covariances.pt &
done

wait

echo "=== RTP Evaluation COMPLETE at: $(date) ==="
echo "Results in: $output_dir/rtp_scores.tsv"
cat "$output_dir/rtp_scores.tsv"
