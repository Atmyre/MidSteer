#!/usr/bin/env bash

sbatch --gpus=4 --job-name=midsteer_sa_fix_50k_last_no_renorm_no_clip_all_strengths ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
sbatch --gpus=4 --job-name=midsteer_sa_fix_50k_last_no_renorm_clip_all_strengths ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping

sbatch --gpus=4 --job-name=midsteer_sa_fix_50k_last_no_renorm_no_clip_zmn_all_strengths ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --zero_mu_neutral
sbatch --gpus=4 --job-name=midsteer_sa_fix_50k_last_no_renorm_clip_zmn_all_strengths ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping --zero_mu_neutral

sbatch --gpus=1 --job-name=midsteer_sa_fix_50k_last_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '2.0 2.5 3.0' --renormalize_after_steering
sbatch --gpus=1 --job-name=midsteer_sa_fix_50k_last_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 '2.0 2.5 3.0' --intermediate_clipping --renormalize_after_steering

sbatch --gpus=1 --job-name=midsteer_sa_fix_20k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_20k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_10k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_10k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_5k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_5k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_1k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_1k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_500_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 500 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_500_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 500 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_100_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 100 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_100_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 100 last 1 '2.0 2.5 3.0' --intermediate_clipping

sbatch --gpus=1 --job-name=midsteer_sa_fix_0_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 0 last 1 '2.0 2.5 3.0'
sbatch --gpus=1 --job-name=midsteer_sa_fix_0_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 0 last 1 '2.0 2.5 3.0' --intermediate_clipping
