#!/usr/bin/env bash

sbatch --job-name=midsteer_sa_5k_last_alpaca_sys_prompt_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_10k_last_alpaca_sys_prompt_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 --use_alpaca_system_prompt
sbatch --job-name=midsteer_sa_20k_last_alpaca_sys_prompt_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1 --use_alpaca_system_prompt

sbatch --job-name=midsteer_sa_5k_last_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 5000 last 1
sbatch --job-name=midsteer_sa_10k_last_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1
sbatch --job-name=midsteer_sa_20k_last_v5 ./llm_exp/sh/slurm_base_experiment.sh self_attn 20000 last 1
