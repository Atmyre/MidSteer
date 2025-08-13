#! /usr/bin/env bash
#SBATCH --partition=camera-long
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./llm_exp/logs/slurm-%x-%j.out
#SBATCH --error=./llm_exp/logs/slurm-%x-%j.err

set -eoux pipefail

export PYTHONPATH=.
python=../miniconda3/bin/python


$python generate_casteer.py --model sana --prompt_file ./test_prompts/snoopy_prompts.txt --steer_type casteer --beta 1 --steer_back --mu_pos ./ckpt/sana/snoopy_imgnt/pos_means_10.pickle --mu_neg ./ckpt/sana/snoopy_imgnt/neg_means_10.pickle --output results/sana/snoopy/ 
# $python generate_casteer.py --model flux-schnell --prompt_file ./test_prompts/snoopy_prompts.txt --steer_type casteer --beta 2 --steer_back --mu_pos ./ckpt/flux-schnell/snoopy_imgnt/pos_means_10.pickle --mu_neg ./ckpt/flux-schnell/snoopy_imgnt/neg_means_10.pickle --output results/flux-schnell/snoopy/ 
