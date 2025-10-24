#!/usr/bin/env bash


qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N teaser_erasure_midsteer_diffusion_sana_10k ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 10000 all relaion '1.0 2.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N teaser_flipping_midsteer_diffusion_sana_10k ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 10000 all relaion '1.0 2.0' 

qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N teaser_erasure_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all relaion '1.0 2.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N teaser_flipping_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all relaion '1.0 2.0' 
