#!/usr/bin/env bash



# sbatch --gpus=8 --job-name=erasure_sana_50k_06 ./exp/sh/slurm_diffusion_erasure.sh sana-06 attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0' 
# sbatch --gpus=8 --job-name=flipping_sana_50k_06 ./exp/sh/slurm_diffusion_flipping.sh sana-06 attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0' 
# sbatch --gpus=8 --job-name=erasure_sana_50k_clip_06 ./exp/sh/slurm_diffusion_erasure.sh sana-06 attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0'  --intermediate_clipping
# sbatch --gpus=8 --job-name=flipping_sana_50k_clip_06 ./exp/sh/slurm_diffusion_flipping.sh sana-06 attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping

# sbatch --gpus=8 --job-name=erasure_sana_50k ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0'
# sbatch --gpus=8 --job-name=flipping_sana_50k ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0'
# sbatch --gpus=8 --job-name=erasure_sana_50k_clip ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0'  --intermediate_clipping
# sbatch --gpus=8 --job-name=flipping_sana_50k_clip ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 50000 all relaion '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping

# sbatch --gpus=4 --job-name=inet_erasure_midsteer_diffusion_sana_50k ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 50000 all imagenet '1.0 2.0 3.0 4.0 5.0' 
# sbatch --gpus=8 --job-name=inet_flipping_midsteer_diffusion_sana_50k ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 50000 all imagenet '1.0 2.0 3.0 4.0 5.0' 
# sbatch --gpus=4 --job-name=inet_erasure_midsteer_diffusion_sana_50k_clip ./exp/sh/slurm_diffusion_erasure.sh sana attn_output 50000 all imagenet '1.0 2.0 3.0 4.0 5.0'  --intermediate_clipping
# sbatch --gpus=8 --job-name=inet_flipping_midsteer_diffusion_sana_50k_clip ./exp/sh/slurm_diffusion_flipping.sh sana attn_output 50000 all imagenet '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping



sbatch --gpus=8 --job-name=coco_erasure_sdxl_50k ./exp/sh/slurm_diffusion_erasure_coco.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 
sbatch --gpus=8 --job-name=coco_flipping_sdxl_50k ./exp/sh/slurm_diffusion_flipping_coco.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 

sbatch --gpus=8 --job-name=coco_erasure_sdxl_50k_clip ./exp/sh/slurm_diffusion_erasure_coco.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0' --intermediate_clipping
sbatch --gpus=8 --job-name=coco_flipping_sdxl_50k_clip ./exp/sh/slurm_diffusion_flipping_coco.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0' --intermediate_clipping


# sbatch --gpus=4 --job-name=erasure_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0'   --intermediate_clipping
# sbatch --gpus=8 --job-name=flipping_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all relaion '1.0 1.5 2.0 2.5 3.0 4.0 5.0'  --intermediate_clipping


# sbatch --gpus=4 --job-name=inet_erasure_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all imagenet '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 
# sbatch --gpus=8 --job-name=inet_flipping_midsteer_diffusion_sdxl_50k ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all imagenet '1.0 1.5 2.0 2.5 3.0 4.0 5.0' 
# sbatch --gpus=4 --job-name=inet_erasure_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_erasure.sh sdxl attn_output 50000 all imagenet '1.0 1.5 2.0 2.5 3.0 4.0 5.0'   --intermediate_clipping
# sbatch --gpus=8 --job-name=inet_flipping_midsteer_diffusion_sdxl_50k_clip ./exp/sh/slurm_diffusion_flipping.sh sdxl attn_output 50000 all imagenet '1.0 1.5 2.0 2.5 3.0 4.0 5.0'  --intermediate_clipping
# 


#sbatch --gpus=8 --job-name=midsteer_diffusion_sana_50k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sana attn_output 50000 all '1.0 2.0 3.0' --intermediate_clipping
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping 
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_renorm ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --renormalize_after_steering
#sbatch --gpus=8 --job-name=midsteer_diffusion_sdxl_50k_all_strengths_clip_renorm ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 50000 all '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping --renormalize_after_steering
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_20k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 20000 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_20k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 20000 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_10k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 10000 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_10k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 10000 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_5k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 5000 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_5k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 5000 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_1k ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 1000 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_1k_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 1000 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_500 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 500 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_500_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 500 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_100 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 100 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_100_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 100 all '2.0 2.5 3.0' --intermediate_clipping 
# 
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_0 ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 0 all '2.0 2.5 3.0'
# sbatch --gpus=4 --job-name=midsteer_diffusion_sdxl_0_clip ./exp/sh/slurm_diffusion_base_experiment.sh sdxl attn_output 0 all '2.0 2.5 3.0' --intermediate_clipping 
