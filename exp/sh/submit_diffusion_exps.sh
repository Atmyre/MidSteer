#!/usr/bin/env bash

sbatch --gpus=4 --job-name=erasure_midsteer_diffusion_sdxl_50k_all_strengths ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 10000 average '1.0 2.0 3.0'
#sbatch --gpus=8 --job-name=flipping_midsteer_diffusion_sdxl_50k_all_strengths ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 10000 average '1.0 2.0 3.0'
sbatch --gpus=4 --job-name=erasure_midsteer_diffusion_sdxl_50k_all_strengths_clip ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 10000 average '1.0 2.0 3.0' --intermediate_clipping
#sbatch --gpus=8 --job-name=flipping_midsteer_diffusion_sdxl_50k_all_strengths_clip ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 10000 average '1.0 2.0 3.0' --intermediate_clipping
#sbatch --gpus=8 --job-name=midsteer_diffusion_sana_50k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sana attn_output 50000 average '1.0 2.0 3.0' --intermediate_clipping
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping 
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_renorm ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --renormalize_after_steering
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_clip_renorm ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 average '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping --renormalize_after_steering
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_20k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 20000 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_20k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 20000 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_10k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 10000 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_10k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 10000 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_5k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 5000 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_5k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 5000 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_1k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 1000 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_1k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 1000 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_500 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 500 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_500_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 500 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_100 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 100 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_100_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 100 average '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_0 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 0 average '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_0_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 0 average '2.0 2.5 3.0' --intermediate_clipping 
