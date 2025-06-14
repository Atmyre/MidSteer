#!/usr/bin/env bash


sbatch --job-name=midsteer_sa_1k_last_retest_with_generation ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 last 1
sbatch --job-name=midsteer_sa_5k_last_retest_with_generation ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1
sbatch --job-name=midsteer_sa_10k_last_retest_with_generation ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1
sbatch --job-name=midsteer_sa_20k_last_retest_with_generation ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1
sbatch --job-name=midsteer_sa_50k_last_retest_with_generation ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1

sbatch --job-name=midsteer_sa_id_all_zero_mu_neutral_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 0 all

sbatch --job-name=midsteer_sa_1k_last_alpaca_sys_prompt_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_5k_last_alpaca_sys_prompt_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_10k_last_alpaca_sys_prompt_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_20k_last_alpaca_sys_prompt_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_50k_last_alpaca_sys_prompt_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 last 1 --use_alpaca_system_prompt

sbatch --job-name=midsteer_sa_1k_all_gen50_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 1000 all 50
sbatch --job-name=midsteer_sa_5k_all_gen50_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 all 50
sbatch --job-name=midsteer_sa_10k_all_gen50_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 all 50
sbatch --job-name=midsteer_sa_20k_all_gen50_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 all 50
sbatch --job-name=midsteer_sa_50k_all_gen50_v2 ./llm_exp/sh/slurm_base_experiment.sh self_attn 50000 all 50
