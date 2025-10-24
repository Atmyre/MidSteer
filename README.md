# MidSteer: Optimal Affine Framework for Steering Generative Models

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **"MIDSTEER: Optimal Affine Framework for Steering Generative Models"** (ICLR 2026).

## Overview

MidSteer is a theoretically-grounded framework for steering generative models (both LLMs and diffusion models) to control their behavior with minimal side effects. Our method:

- **Bridges theory and practice**: We show that vanilla steering is a special case of LEACE (concept erasure), and introduce MidSteer as an optimal framework for concept switching
- **Supports multiple tasks**: Concept erasure (removing unwanted behaviors) and concept flipping (switching one concept to another)
- **Works across modalities**: Applicable to both Large Language Models and image diffusion models
- **Zero inference overhead**: Can be incorporated directly into model weights

## Key Results

- **Better concept switching**: MidSteer achieves superior balance between flipping desired concepts while preserving unrelated features
- **Minimal disturbance**: Preserves model quality on unrelated tasks (MMLU for LLMs, unrelated concepts for diffusion)
- **Theoretical guarantees**: Provides closed-form solutions to optimal affine steering problems

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for LLM experiments, 16GB+ for diffusion)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/midsteer.git
cd midsteer
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# For macOS/Darwin
pip install -r requirements/darwin.txt

# For Linux
pip install -r requirements/linux.txt
```

4. Set up Hugging Face authentication (required for downloading models):
```bash
huggingface-cli login
```

## Repository Structure

```
midsteer/
├── core/                       # Core library code
│   ├── llm_steering.py        # LLM steering implementation
│   ├── diffusion_steering.py  # Diffusion model steering
│   ├── controller.py          # Vector control logic
│   ├── dataset.py             # Dataset utilities
│   └── eval/                  # Evaluation metrics (CLIP, FID)
├── scripts/
│   ├── llm/                   # LLM experiment scripts
│   │   ├── generate_steering_vectors.py
│   │   ├── estimate_covariances.py
│   │   ├── run_with_steering.py
│   │   ├── concept_scoring.py
│   │   └── consistency_scoring.py
│   └── diffusion/             # Diffusion model scripts
│       ├── estimate_steering_vectors.py
│       ├── estimate_covariances.py
│       ├── run_with_steering.py
│       └── produce_scores.py
├── exp/
│   ├── datasets/              # Training and evaluation datasets
│   │   ├── train/            # Concept-specific questions for LLMs
│   │   └── eval/             # Evaluation templates
│   └── sh/                   # Shell scripts for running experiments
├── notebooks/
│   └── produce_charts.ipynb  # Generate paper figures
└── requirements/             # Platform-specific requirements
```

## Reproducing Paper Results

Our paper presents experiments on two main tasks across multiple models. Here's how to reproduce each result:

### 1. LLM Experiments

#### Models Tested
- Llama-2-7B-chat-hf
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct

#### Concept Pairs
- horses ↔ motorcycles
- dogs ↔ cats

#### A. Concept Erasure (Section 4.2.1)

This experiment removes unwanted concepts from LLM outputs.

**Quick start (single GPU):**
```bash
.venv/bin/python3 scripts/llm/estimate_covariances.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --token_aggregation_mode all \
    --num_samples 50000 \
    --max_new_tokens 100 \
    --output_dir ./results/llama2-7b/covariances

.venv/bin/python3 scripts/llm/generate_steering_vectors.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --topics horses motorcycles dogs cats \
    --token_aggregation_mode last \
    --max_new_tokens 1 \
    --num_samples 1000 \
    --output_dir ./results/llama2-7b/steering_vectors

.venv/bin/python3 scripts/llm/run_with_steering.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --source_concept horses \
    --source_concept_path ./results/llama2-7b/steering_vectors/horses.pt \
    --target_concept_path ./results/llama2-7b/steering_vectors/motorcycles.pt \
    --steer_type midsteer \
    --strength 1.0 \
    --mu_neutral ./results/llama2-7b/covariances/means.pt \
    --cov_neutral ./results/llama2-7b/covariances/covariances.pt \
    --dataset_type template \
    --samples_per_question 10 \
    --max_new_tokens 100 \
    --output_dir ./results/llama2-7b/evaluation/horses_erasure

# Score the results
.venv/bin/python3 scripts/llm/concept_scoring.py \
    --concept horses motorcycles \
    --dir ./results/llama2-7b/evaluation/horses_erasure

.venv/bin/python3 scripts/llm/consistency_scoring.py \
    --dir ./results/llama2-7b/evaluation/horses_erasure
```

**Full experiment with multiple methods and strengths:**
```bash
# For SLURM clusters
sbatch --job-name=llm-erasure-llama2 \
    exp/sh/slurm_llm_base_experiment.sh \
    meta-llama/Llama-2-7b-chat-hf \
    self_attn \
    50000 \
    all \
    100 \
    "0.5 1.0 1.5 2.0 2.5 3.0"

# For Grid Engine clusters (qsub)
qsub -N llm-erasure-llama2 \
    exp/sh/slurm_llm_base_experiment.sh \
    meta-llama/Llama-2-7b-chat-hf \
    self_attn \
    50000 \
    all \
    100 \
    "0.5 1.0 1.5 2.0 2.5 3.0"
```

This script will:
1. Estimate covariances from 50k neutral prompts (Alpaca dataset)
2. Generate steering vectors for each concept (horses, motorcycles, dogs, cats)
3. Run experiments with CASteer, LEACE, and MidSteer at various strengths
4. Evaluate on template prompts, MMLU, and Alpaca datasets
5. Compute concept scores and consistency metrics

**Output:** Results are saved to `exp/results/{model_name}/{job_name}/evaluation/`

#### B. Concept Flipping (Section 4.2.2)

Same as erasure, but switches one concept to another (e.g., horses → motorcycles).

The `slurm_llm_base_experiment.sh` script runs both erasure and flipping experiments. Results for flipping appear in directories like `horses_to_motorcycles__horses/`.

### 2. Diffusion Model Experiments

#### Models Tested
- Stable Diffusion XL (SDXL)
- SANA 1.6B

#### Concept Pairs
- horse ↔ motorcycle
- snoopy ↔ mickey
- chihuahua ↔ muffin

#### A. Concept Erasure & Flipping

**Quick start (single GPU):**
```bash
.venv/bin/python3 scripts/diffusion/estimate_covariances.py \
    --model_name sdxl-turbo \
    --control_mode attn_output \
    --aggregation_mode all \
    --num_samples 50000 \
    --output_dir ./results/sdxl/covariances

.venv/bin/python3 scripts/diffusion/estimate_steering_vectors.py \
    --model_name sdxl-turbo \
    --control_mode attn_output \
    --topics horse motorcycle snoopy mickey chihuahua muffin \
    --aggregation_mode average \
    --num_samples 1000 \
    --output_dir ./results/sdxl/steering_vectors

.venv/bin/python3 scripts/diffusion/run_with_steering.py \
    --model_name sdxl \
    --control_mode attn_output \
    --generate_concept horse \
    --output_dir ./results/sdxl/evaluation/horse_to_motorcycle/midsteer-1.0 \
    --steering_method midsteer \
    --steering_strength 1.0 \
    --covariances_dir ./results/sdxl/covariances \
    --num_images_per_prompt 10 \
    --seed 42 \
    translate \
    --source_concept_path ./results/sdxl/steering_vectors/horse.pt \
    --target_concept_path ./results/sdxl/steering_vectors/motorcycle.pt

# Compute CLIP scores and FID
.venv/bin/python3 scripts/diffusion/produce_scores.py \
    --concept horse motorcycle \
    --dir ./results/sdxl/evaluation/horse_to_motorcycle \
    --num_workers 4 \
    --batch_size 32
```

**Full experiment with all methods:**
```bash
# For SLURM clusters
sbatch --job-name=diffusion-sdxl \
    exp/sh/slurm_diffusion_base_experiment.sh \
    sdxl \
    attn_output \
    50000 \
    all \
    "0.5 1.0 1.5 2.0 2.5 3.0"

# For Grid Engine
qsub -N diffusion-sdxl \
    exp/sh/slurm_diffusion_base_experiment.sh \
    sdxl \
    attn_output \
    50000 \
    all \
    "0.5 1.0 1.5 2.0 2.5 3.0"
```

This runs comprehensive experiments including:
- Concept translation (flipping) for all concept pairs
- Concept erasure for all concepts
- Multiple steering strengths with CASteer, LEACE, and MidSteer
- CLIP score and FID computation

**Output:** Results saved to `exp/results/{model_name}/{job_name}/evaluation/`

### 3. Generating Paper Figures

After running experiments, generate the Pareto frontier plots and tables from the paper:

```bash
jupyter lab notebooks/produce_charts.ipynb
```

This notebook:
- Loads results from experiment directories
- Computes Pareto frontiers for each method
- Generates plots comparing CASteer, LEACE, and MidSteer
- Produces tables with numerical results
- Exports figures to `artefacts/` directory

### 4. Expected Results

**LLM Concept Flipping (horses → motorcycles):**
- MidSteer: Successfully switches concepts while preserving "motorcycle" prompt integrity
- LEACE/CASteer: May affect both forward and reverse directions

**Diffusion Concept Flipping:**
- MidSteer: Better preservation of unrelated concepts (lower FID on unrelated concepts)
- Higher CLIP score difference between source and target concepts

**Key Metrics:**
- **CS (Concept Score)**: Relevance to target concept (0-10 for LLM, CLIP score for diffusion)
- **ΔCS**: Difference between target and source concept scores
- **BERT Score**: Consistency of generated text (LLM only)
- **FID**: Image quality preservation (diffusion only)

## Advanced Usage

### Custom Concepts

1. **For LLMs**: Create concept-specific question files in `exp/datasets/train/`:
```bash
.venv/bin/python3 helpers/generate_llm_training_questions.py \
    --concept "your_concept" \
    --num_samples 1000 \
    --output_dir exp/datasets/train
```

2. **For Diffusion**: Concepts are specified as simple strings (e.g., "robot", "castle")

### Steering Methods

Three methods are available via `--steer_type` / `--steering_method`:

- `casteer`: Vanilla steering (baseline)
- `leace`: LEACE (optimal concept erasure)
- `midsteer`: MidSteer (our method, optimal concept switching)

### Steering Strength

The `--strength` parameter controls steering intensity:
- **Erasure**: β = 1.0 is optimal for LEACE
- **Flipping**: β = 2.0 for CASteer/LEACE, β = 1.0 for MidSteer (as per theory)
- Experiment with values 0.5-5.0 to find best trade-offs

### Optional Flags

- `--intermediate_clipping`: Clip dot product to ≥0 (non-matrix form)
- `--mm_normalize_centers`: Normalize class-conditional means
- `--renormalize_after_steering`: Renormalize activations after steering
- `--zero_mu_neutral`: Use zero vector instead of estimated mean
- `--identity_cov`: Use identity covariance (equivalent to num_covariances=0)

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{midsteer2026,
  title={MIDSTEER: Optimal Affine Framework for Steering Generative Models},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Hardware Requirements

### Minimum Requirements
- **LLM experiments**: 
  - 1x GPU with 24GB VRAM (for 7B models)
  - 2x GPUs with 40GB VRAM each (for 13B+ models)
- **Diffusion experiments**:
  - 1x GPU with 16GB VRAM (SDXL)
  - 1x GPU with 24GB VRAM (SANA)

### Recommended for Full Reproduction
- Multi-GPU cluster for parallel experiment execution
- Storage: ~100GB for cached models, ~50GB for experiment results

### Computational Time Estimates
- Covariance estimation: 2-4 hours (50k samples, single GPU)
- Steering vector generation: 30 minutes (1k samples per concept)
- Single steering run: 5-10 minutes (LLM), 10-20 minutes (diffusion)
- Full experiment (all methods, strengths): 8-12 hours per model

## Troubleshooting

### Out of Memory
- Reduce batch size in generation scripts
- Use gradient checkpointing (enabled by default)
- Reduce `num_samples` for covariance estimation (minimum ~5k)

### Slow Generation
- Use turbo variants: `sdxl-turbo` instead of `sdxl`
- Reduce `samples_per_question` or `num_images_per_prompt`
- Enable multi-GPU with SLURM scripts

### Missing Dependencies
- Ensure correct requirements file for your platform
- Install clean-fid separately if issues: `pip install clean-fid`
- For CLIP issues: `pip install git+https://github.com/openai/CLIP.git`

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- LEACE framework from [Belrose et al., 2023]
- CASteer approach from [Gaintseva et al., 2024]
- Steering vectors library: [steering-vectors](https://github.com/nrimsky/steering-vectors)

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
