#!/usr/bin/env bash
# Step 3+4 for SANA nudity experiment — sequential per concept, parallel within concept
set -eoux pipefail

cd /root/midsteer
export PYTHONPATH=/root/midsteer

BASE=exp/results/sana/nudity_erasure
SV=$BASE/steering_vectors
COV=$BASE/covariances
RESULTS=$BASE/evaluation
TEMPLATE=exp/datasets/eval/concepts/template_nudity.json

SOURCE=nudity
TARGET=clothed
STRENGTHS="1.0 2.0 3.0 4.0 5.0"

run_concept() {
    local concept=$1
    local gpu=$2
    local subdir=$RESULTS/concept_translation/${SOURCE}_to_${TARGET}__${concept}
    mkdir -p $subdir

    local common="--model_name sana --control_mode attn_output --num_images_per_prompt 5 --seed 42 --template_path $TEMPLATE --file_format JPEG --covariances_dir $COV"

    # Baseline
    CUDA_VISIBLE_DEVICES=$gpu python scripts/diffusion/run_with_steering.py \
        --generate_concept "$concept" --output_dir "$subdir/orig" $common

    for method in casteer leace midsteer; do
        for strength in $STRENGTHS; do
            CUDA_VISIBLE_DEVICES=$gpu python scripts/diffusion/run_with_steering.py \
                --generate_concept "$concept" --output_dir "$subdir/$method-$strength" \
                --steering_method $method --steering_strength $strength \
                $common \
                translate --source_concept_path $SV/$SOURCE.pt --target_concept_path $SV/$TARGET.pt
        done
    done

    echo "DONE: $concept on GPU $gpu"
}

echo "=== Step 3: Image generation ==="
echo "Started at: $(date)"

# Run concepts in pairs (2 GPUs)
concepts=(woman man girl boy couple person cat car landscape building)

for ((i=0; i<${#concepts[@]}; i+=2)); do
    c1=${concepts[$i]}
    c2=${concepts[$((i+1))]-}

    run_concept "$c1" 0 &
    if [ -n "$c2" ]; then
        run_concept "$c2" 1 &
    fi
    wait
    echo "Batch done: $c1 $c2 at $(date)"
done

echo "=== Step 3 COMPLETE at: $(date) ==="

# Step 4: CLIP + FID scoring
echo "=== Step 4: Scoring ==="
for concept in "${concepts[@]}"; do
    subdir=$RESULTS/concept_translation/${SOURCE}_to_${TARGET}__${concept}
    python scripts/diffusion/produce_scores.py \
        --concept $SOURCE $TARGET $concept \
        --dir $subdir \
        --num_workers 4 --batch_size 32 &
done
wait

echo "=== Step 4 COMPLETE at: $(date) ==="
echo "=== ALL DONE ==="
