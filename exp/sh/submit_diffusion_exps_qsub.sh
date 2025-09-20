#!/usr/bin/env bash


qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_erasure_midsteer_diffusion_sana_10k ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 10000 all '1.0 2.0 3.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_flipping_midsteer_diffusion_sana_10k ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 10000 all '1.0 2.0 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_erasure_midsteer_diffusion_sana_10k_clip ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 10000 all '1.0 2.0 3.0'  --intermediate_clipping
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_flipping_midsteer_diffusion_sana_10k_clip ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 10000 all '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping

qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_erasure_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_flipping_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_erasure_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 4.0 5.0'   --intermediate_clipping
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N inet_flipping_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 4.0 5.0'  --intermediate_clipping
