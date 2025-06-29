#!/usr/bin/env bash

sbatch --job-name=midsteer_sa_10k_last ./llm_exp/sh/slurm_base_experiment.sh self_attn 10000 last 1 

