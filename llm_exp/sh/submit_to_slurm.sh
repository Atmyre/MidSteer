#!/usr/bin/env bash

sbatch --job-name=midsteer_sa_10k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1
sbatch --job-name=midsteer_sa_10k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 --intermediate_clipping
sbatch --job-name=midsteer_sa_10k_last_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 --renormalize_after_steering
sbatch --job-name=midsteer_sa_10k_last_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 --renormalize_after_steering --intermediate_clipping


sbatch --job-name=midsteer_sa_20k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1
sbatch --job-name=midsteer_sa_20k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 --intermediate_clipping
sbatch --job-name=midsteer_sa_20k_last_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 --renormalize_after_steering
sbatch --job-name=midsteer_sa_20k_last_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 --renormalize_after_steering --intermediate_clipping

sbatch --job-name=midsteer_sa_50k_last_no_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1
sbatch --job-name=midsteer_sa_50k_last_no_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 --intermediate_clipping
sbatch --job-name=midsteer_sa_50k_last_renorm_no_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 --renormalize_after_steering
sbatch --job-name=midsteer_sa_50k_last_renorm_clip ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 --renormalize_after_steering --intermediate_clipping
