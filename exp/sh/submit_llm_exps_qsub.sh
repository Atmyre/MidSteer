#!/usr/bin/env bash


qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N midsteer_sa_fix_50k_last_no_renorm_no_clip_all_strengths ./exp/sh/slurm_llm_base_experiment.sh 'meta-llama/Llama-2-7b-chat-hf' self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0'
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N midsteer_sa_fix_50k_last_no_renorm_clip_all_strengths ./exp/sh/slurm_llm_base_experiment.sh 'meta-llama/Llama-2-7b-chat-hf' self_attn 50000 last 1 '1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0' --intermediate_clipping


qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N erasure_qwen_7b_50k ./exp/sh/slurm_llm_erasure_experiment.sh 'Qwen/Qwen2.5-7B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N erasure_qwen_14b_50k ./exp/sh/slurm_llm_erasure_experiment.sh 'Qwen/Qwen2.5-14B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N llm_flipping_midsteer_qwen_50k ./exp/sh/slurm_llm_base_experiment.sh 'Qwen/Qwen2.5-7B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' 
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N llm_flipping_midsteer_qwen_14b_50k ./exp/sh/slurm_llm_base_experiment.sh 'Qwen/Qwen2.5-14B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' 

qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N llm_flipping_midsteer_qwen_50k_clip ./exp/sh/slurm_llm_base_experiment.sh 'Qwen/Qwen2.5-7B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N erasure_qwen_7b_50k_clip ./exp/sh/slurm_llm_erasure_experiment.sh 'Qwen/Qwen2.5-7B-Instruct' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping
qsub -pe smp 32 -l gpu=4,h_vmem=7.5G -N midsteer_erasure_llama2_clip ./exp/sh/slurm_llm_erasure_experiment.sh 'meta-llama/Llama-2-7b-chat-hf' self_attn 50000 last 1 '1.0 2.0 3.0 4.0 5.0' --intermediate_clipping
