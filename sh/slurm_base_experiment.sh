#! /usr/bin/env bash
#SBATCH --partition=camera-long
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./llm_exp/logs/slurm-%x-%j.out
#SBATCH --error=./llm_exp/logs/slurm-%x-%j.err

set -eoux pipefail

export PYTHONPATH=.
python=../miniconda3/bin/python

# $python ./compute_steering_vectors.py --model sana --mode file --prompts_pos_file ./concept_prompts/imagenet_classes_snoopy.txt --prompts_neg_file ./concept_prompts/imagenet_classes.txt --output_dir ckpt/sana/snoopy_imgnt/ --checkpoint_steps 1,10,50,100,500
$python ./compute_steering_vectors.py --model flux-schnell --mode file --prompts_pos_file ./concept_prompts/imagenet_classes_snoopy.txt --prompts_neg_file ./concept_prompts/imagenet_classes.txt --output_dir ckpt/flux-schnell/snoopy_imgnt/ --checkpoint_steps 1,10,50,100,500
