#!/usr/bin/env bash
# Waits for LLM experiment to finish, then runs SDXL and SANA violence/peace experiments
set -eoux pipefail

cd /root/midsteer

echo "Waiting for LLM experiment to finish..."
while ! grep -q "ALL STEPS COMPLETE" /root/midsteer/experiment.log 2>/dev/null; do
    sleep 60
    echo "$(date): LLM experiment still running..."
done
echo "LLM experiment finished at $(date)!"

# Run SDXL first (works with diffusers 0.30.0)
echo "========================================="
echo "Starting SDXL violence/peace experiment"
echo "========================================="
bash exp/sh/run_diffusion_violence_peace_runpod.sh sdxl "1.0 2.0 3.0 4.0 5.0" \
    > /root/midsteer/experiment_sdxl_vp.log 2>&1

echo "SDXL done at $(date)!"

# Run SANA (may need newer diffusers)
echo "========================================="
echo "Starting SANA violence/peace experiment"
echo "========================================="
bash exp/sh/run_diffusion_violence_peace_runpod.sh sana "1.0 2.0 3.0 4.0 5.0" \
    > /root/midsteer/experiment_sana_vp.log 2>&1

echo "SANA done at $(date)!"
echo "ALL DIFFUSION EXPERIMENTS COMPLETE at $(date)"
