#!/usr/bin/env bash


sbatch --job-name=midsteer_sa_id_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 0 all
sbatch --job-name=midsteer_sa_1k_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 all
sbatch --job-name=midsteer_sa_5k_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 all
sbatch --job-name=midsteer_sa_10k_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 all
sbatch --job-name=midsteer_sa_20k_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 all
sbatch --job-name=midsteer_sa_50k_all ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 all

sbatch --job-name=midsteer_sa_1k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 last
sbatch --job-name=midsteer_sa_5k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last
sbatch --job-name=midsteer_sa_10k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last
sbatch --job-name=midsteer_sa_20k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last
sbatch --job-name=midsteer_sa_50k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last
