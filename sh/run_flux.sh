#! /usr/bin/env bash
#SBATCH --partition=camera-long
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./llm_exp/logs/slurm-%x-%j.out
#SBATCH --error=./llm_exp/logs/slurm-%x-%j.err

set -eoux pipefail

export PYTHONPATH=.
python=../miniconda3/bin/python


$python generate_casteer.py --model sana --prompt_file ./test_prompts/mickey_prompts.txt --output results/sana/mickey/ --not_steer

