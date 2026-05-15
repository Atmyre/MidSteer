# MidSteer: Optimal Affine Framework for Steering Generative Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.05220-b31b1b.svg)](https://arxiv.org/abs/2605.05220)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/ICML-2026-blueviolet.svg)](https://icml.cc/Conferences/2026)

Official implementation of **"MidSteer: Optimal Affine Framework for Steering Generative Models"**, accepted at ICML 2026.

> **TL;DR.** Steering the internal representations of an LLM or diffusion model lets you flip one concept to another (e.g. "horse" → "motorcycle", "toxicity" → "helpfulness") without retraining. We prove that the standard *steering vector* trick is a special case of LEACE (closed-form affine concept erasure), generalize it to concept *switching*, and show that the resulting method — MidSteer — preserves unrelated features better than prior heuristics while adding zero inference-time overhead.

**Authors.** Tatiana Gaintseva, Andrew Stepanov, Ziquan Liu, Martin Benning, Gregory Slabaugh, Jiankang Deng, Ismail Elezi.

---

## Citation

```bibtex
@inproceedings{gaintseva2026midsteer,
  title     = {{MidSteer}: Optimal Affine Framework for Steering Generative Models},
  author    = {Gaintseva, Tatiana and Stepanov, Andrew and Liu, Ziquan
               and Benning, Martin and Slabaugh, Gregory and Deng, Jiankang
               and Elezi, Ismail},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.05220}
}
```

A `CITATION.cff` is also provided so GitHub renders a "Cite this repository" widget.

---

## What's in this repo

| Path                          | Purpose                                                    |
| ----------------------------- | ---------------------------------------------------------- |
| `core/`                       | Library code: math, controllers, model loading, datasets   |
| `scripts/llm/`                | LLM experiment entry points (covariances, vectors, eval)   |
| `scripts/diffusion/`          | Diffusion model entry points                               |
| `exp/sh/`                     | Seven portable example scripts mapping to paper sections   |
| `exp/datasets/eval/`          | Evaluation templates and prompts (Alpaca, MMLU, COCO 30k, BeaverTails, abstract concept templates) |
| `exp/datasets/train/`         | Per-concept training questions used to estimate steering vectors |
| `notebooks/produce_charts.ipynb` | Pareto frontier plots and result tables                 |
| `requirements/`               | Platform-specific dependency lists                         |
| `helpers/`                    | Convenience scripts (e.g. generating new training prompts) |

---

## Installation

### Requirements

- Python 3.10 or later (3.13 tested).
- A CUDA-capable GPU. Most experiments target a single H100; SDXL works on 16 GB cards, Llama-2-7B on 24 GB, Qwen2.5-14B and SANA-1.5 on 40 GB+.
- ~100 GB of disk space for cached model weights, ~50 GB more for experiment outputs.

### Setup

```bash
git clone https://github.com/Atmyre/MidSteer.git
cd MidSteer

python3 -m venv .venv
source .venv/bin/activate

# Pick the requirements file matching your OS:
pip install -r requirements/linux.txt   # or requirements/darwin.txt on macOS
```

### Environment variables

The code reads two secrets from the environment. Copy `.env.example` to `.env` and fill them in:

```bash
cp .env.example .env
$EDITOR .env
```

| Variable        | Required? | Used for                                                   |
| --------------- | --------- | ---------------------------------------------------------- |
| `HF_TOKEN`      | Yes       | Downloading gated HuggingFace models (Llama-2, Qwen, SANA, FLUX). Get one at https://huggingface.co/settings/tokens. |
| `OPENAI_API_KEY`| Optional  | Only needed if you run the GPT-4o-mini cross-validation judge (`scripts/llm/concept_scoring_gpt4o.py`). |

If you prefer, `huggingface-cli login` also works in place of setting `HF_TOKEN`.

---

## Quick start

Run the smallest end-to-end demo: estimate the covariance, build steering vectors for two concepts on Llama-2-7B, and generate one steered sample.

```bash
export HF_TOKEN=hf_...

# 1. Neutral-prompt covariance Σ_XX (one-time per model)
python scripts/llm/estimate_covariances.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --token_aggregation_mode all \
    --num_samples 5000 \
    --output_dir ./demo/cov

# 2. Per-concept steering vectors
python scripts/llm/generate_steering_vectors.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --topics horses motorcycles \
    --num_samples 200 \
    --output_dir ./demo/sv

# 3. Generate with MidSteer at strength β=2
python scripts/llm/run_with_steering.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --layer_type self_attn \
    --source_concept horses \
    --source_concept_path ./demo/sv/horses.pt \
    --target_concept_path ./demo/sv/motorcycles.pt \
    --steer_type midsteer \
    --strength 2.0 \
    --mu_neutral  ./demo/cov/means.pt \
    --cov_neutral ./demo/cov/covariances.pt \
    --dataset_type template \
    --samples_per_question 1 \
    --output_dir ./demo/out
```

`./demo/cov` will take ~5 minutes on an H100 with the reduced `num_samples=5000`; the paper's main experiments use 50 000 samples.

---

## Reproducing paper results

Each example script in `exp/sh/` reproduces a result from the paper or rebuttal. They are pure portable bash — no SLURM/qsub directives — and configurable via env vars at the top.

| Script                                       | Paper reference                  | What it does                                              |
| -------------------------------------------- | -------------------------------- | --------------------------------------------------------- |
| `exp/sh/01_llm_concept_switching.sh`         | §5, Tables 1–2                   | Llama-2-7B, horses → motorcycles, 3 methods × 9 strengths |
| `exp/sh/02_diffusion_concept_switching.sh`   | §5, Figs. 3–4                    | SDXL, dogs → cats, 3 methods × 9 strengths                |
| `exp/sh/03_llm_concept_erasure.sh`           | App. J (LLM erasure)             | Llama-2-7B, erase "horses", LEACE vs CASteer              |
| `exp/sh/04_diffusion_concept_erasure.sh`     | App. J (Diffusion erasure)       | SDXL, erase "dog", LEACE vs CASteer                       |
| `exp/sh/05_safety_toxicity_to_helpfulness.sh`| Rebuttal Tabs. 1–2 (LLM safety)  | Llama-2-7B, toxicity → helpfulness, RealToxicityPrompts + Detoxify |
| `exp/sh/06_safety_violence_to_peace.sh`      | Rebuttal Tabs. 3–4 (Diff safety) | SDXL or SANA, violence → peace, CLIP-based scoring         |
| `exp/sh/07_gpt4o_judge_crossval.sh`          | App. I.4                         | Re-scores outputs of script 01 with GPT-4o-mini           |

Run them like:

```bash
bash exp/sh/01_llm_concept_switching.sh

# Override defaults via env vars at the call site:
OUTPUT_DIR=./my_results STRENGTHS="1.0 2.0 3.0" \
    bash exp/sh/01_llm_concept_switching.sh
```

After a script finishes, open `notebooks/produce_charts.ipynb` to compute Pareto frontiers and produce the figures from the paper. The notebook expects results in `./results/<experiment>/evaluation/...`, which is where the scripts write by default.

> **Note on AxBench.** The rebuttal also reports a LEACE-on-Gemma-2-2B evaluation against AxBench (rebuttal Table 1). That experiment is currently **not** part of this repo — running it requires the upstream AxBench harness ([Wu et al., 2025](https://arxiv.org/abs/2501.17148)) and a `gemma` branch in `core/utils.py:init_llm_model_and_tokenizer`, neither of which is shipped here.

---

## Datasets

**Bundled in `exp/datasets/eval/`** (no download needed):

- `alpaca_instruct/alpaca_data.json` — Stanford Alpaca instructions
- `coco/coco_30k.csv` — 30 K captions from MS-COCO 2014
- `mmlu/mmlu_full.json` — MMLU questions
- `concepts/{horses,dogs,cats}.json` — paper's per-concept template prompts
- `concepts/template.json` — default fill-in-the-blank template
- `concepts/template_{abstract,toxicity_helpfulness,violence_peace,nudity}.json` — templates for the rebuttal experiments
- `concepts/beavertails_eval.json` — 700 BeaverTails-Evaluation red-team prompts
- `imagenet/template.json` — ImageNet class-name templates

**Auto-downloaded from HuggingFace at runtime**:

- `laion/relaion2B-en-research` — used by `scripts/diffusion/estimate_covariances.py` and `estimate_steering_vectors.py`
- `allenai/real-toxicity-prompts` — used by `scripts/llm/run_rtp_eval.py`

**Training prompts** in `exp/datasets/train/*.json` are pre-generated batches of concept-specific questions used to estimate the class-conditional means in each steering vector. To make new ones:

```bash
python scripts/llm/generate_training_questions.py \
    --topics my_concept \
    --output_dir exp/datasets/train
```

---

## Hardware and runtime

| Step                              | Memory   | Wall time (H100, single GPU)              |
| --------------------------------- | -------- | ----------------------------------------- |
| LLM covariance (50 K samples)     | ~24 GB   | ~15 min Llama-2-7B, ~30 min Qwen2.5-14B   |
| LLM steering vectors per concept  | ~24 GB   | ~1 min for 1000 samples                   |
| LLM steered generation per config | ~24 GB   | ~5–10 min per (method, strength, concept) |
| SDXL covariance (50 K)            | ~16 GB   | ~15 min (using `sdxl-turbo`)              |
| SDXL generation per config        | ~16 GB   | ~10–20 min                                |
| SANA covariance / generation      | ~24 GB   | comparable to SDXL                        |
| Full `01_llm_concept_switching`   | 1 GPU    | ~12 h sequential                          |
| Full `02_diffusion_concept_switching` | 1 GPU | ~18 h sequential                          |

To parallelize across GPUs, append `&` inside the inner loops of any example script and add `wait` after each phase.

---

## Troubleshooting

- **"401 Unauthorized" from HuggingFace.** `HF_TOKEN` is unset or doesn't have access to the gated model (Llama-2 and Gemma both require accepting a license once on the HF website).
- **`SanaPipeline` not found.** Install diffusers ≥ 0.32: `pip install -U diffusers`.
- **`clean-fid` install fails.** Some platforms need it separately: `pip install clean-fid`.
- **OOM during covariance estimation.** Drop `--num_samples` to ~5000; the rebuttal shows that the metric stabilizes well below 50 K.
- **`bert-score` cold start is slow.** The first call downloads RoBERTa-large; subsequent calls hit the cache.

---

## License

MIT — see `LICENSE`.

---
<!-- ────────────────────────────────────────────────────────────────────── -->

## For AI assistants and contributors

This section is for AI coding assistants and human contributors who want to extend or modify the codebase. It documents internal abstractions, file roles, and extension points that are not obvious from skimming the code.

### Code layout, drill-down

```
core/
├── controller.py         VectorControl (ABC) + CrossAttentionOutputSteering
│                         — the runtime hook that intercepts intermediate
│                           activations and rewrites them in place.
├── math.py               fractional_matrix_power_cov_torch (whitening),
│                         dtype helpers. The closed-form solutions from
│                         the paper's Theorems 3.3 / 4.2 / 4.4 reduce to
│                         calls into this module.
├── llm_steering.py       LLM-specific Controller wiring (hooks into
│                         self_attn / mlp / decoder_block / etc.).
├── diffusion_steering.py Diffusion-specific wiring (UNet cross-attn).
├── utils.py              init_image_model_and_tokenizer, init_llm_model_and_tokenizer
│                         — single source of truth for "which HF model ID
│                           and dtype maps to which keyword".
├── dataset.py            QuestionsDataset hierarchy: TemplateDataset,
│                         AlpacaDataset, MMLUDataset, RelaionDataset,
│                         ImageNetDataset, CocoDataset.
├── prompt_utils.py       Llama-chat tokenization helpers.
├── prompts.py            Prompt templates for the LLM-as-judge.
├── pickle.py             Defensive loading: torch.load + pickle fallback
│                         + a CPU_Unpickler that strips CUDA tensors.
├── vector_dump.py        Tensor I/O helpers.
└── eval/
    ├── clip.py           CLIP-based concept score for diffusion outputs.
    └── fid.py            FID via clean-fid.
```

### Core abstractions

**`VectorControl`** (`core/controller.py:32`) is the ABC every steering method implements. Its `__call__` is invoked by the model's forward hook once per (diffusion step × cross-attention layer × token position). It returns a tensor of the same shape, optionally modified.

The concrete class **`CrossAttentionOutputSteering`** (line 80) applies the paper's affine transform:

- For each tracked layer, the controller stores per-concept *steering vectors* (the centred class-conditional means) and the global covariance `Σ_XX`.
- At inference, the activation `x` is mapped to `A·x + b` where `A` and `b` are determined by `--steer_type` (`casteer`, `leace`, or `midsteer`) and the strength `β`.
- The math lives in `core/math.py` and is invoked through `fractional_matrix_power_cov_torch` (which whitens `Σ_XX`).

**`SteeringVectors`** is the type alias

```python
SteeringVectors = dict[int, dict[str, list[torch.Tensor]]]
#                  ^step ^place-in-network ^per-layer
```

For LLMs there's exactly one diffusion step (key `0`). For diffusion models, vectors can vary per step, and the controller advances `_diffusion_step` automatically.

### Pickled artefacts

Every script writes pickled `torch.Tensor` containers via `core/pickle.py`:

| File written by                              | Contents                                |
| -------------------------------------------- | --------------------------------------- |
| `estimate_covariances.py` → `means.pt`       | `dict[step, dict[place, list[Tensor]]]` of per-layer neutral means `µ_XX` |
| `estimate_covariances.py` → `covariances.pt` | Same structure, holds per-layer `Σ_XX`  |
| `generate_steering_vectors.py` → `<topic>.pt`| Per-concept class-conditional mean (same structure) |

When loading, `core/pickle.unpickle()` tolerates pickles produced on a different machine (it remaps CUDA storage to CPU). Use `unpickle_pack()` to concat multiple files passed as a comma-separated path.

### Adding a new model

Both initializers in `core/utils.py` are big `if/elif` chains. Each branch declares the HuggingFace ID, dtype, and any model-specific kwargs (e.g. CPU offloading for FLUX).

To add support for, say, Gemma-2-2B:

1. Add a branch to `init_llm_model_and_tokenizer` at `core/utils.py:317`:

   ```python
   elif 'gemma' in model_name.lower():
       model = AutoModelForCausalLM.from_pretrained(
           model_name,
           cache_dir=cache_dir,
           torch_dtype=torch.bfloat16,
           device_map='balanced',
           token=os.getenv("HF_TOKEN"),
       )
       tokenizer = AutoTokenizer.from_pretrained(
           model_name, cache_dir=cache_dir, token=os.getenv("HF_TOKEN")
       )
       return model, tokenizer
   ```

2. Check whether `core/llm_steering.py` needs a new layer-type mapping — Gemma uses RMSNorm so the `self_attn` path should "just work", but verify by running `01_llm_concept_switching.sh` against the new model with a small `NUM_COV_SAMPLES`.

For diffusion models the pattern is the same in `init_image_model_and_tokenizer` (same file).

### Adding a new concept

1. Add a template file in `exp/datasets/eval/concepts/<my_concept>.json`. It is a JSON list of strings; the loader replaces the literal `{}` with the concept name. Look at `template_violence_peace.json` for a working example.
2. Generate training prompts (used to estimate the class-conditional mean):

   ```bash
   python scripts/llm/generate_training_questions.py \
       --topics my_concept \
       --output_dir exp/datasets/train
   ```

3. Run one of the example scripts (`01_llm_concept_switching.sh` is the closest fit), changing the `--topics` / `--source_concept` / `--target_concept` arguments.

### Adding a new steering method

The dispatch happens in `core/controller.py`'s `CrossAttentionOutputSteering.__init__` based on `steer_type`. To add a method:

1. Add a string to the choices in both `scripts/llm/run_with_steering.py` and `scripts/diffusion/run_with_steering.py` (`choices=['casteer', 'leace', 'midsteer']`).
2. Implement a new branch inside `CrossAttentionOutputSteering` that computes `A` and `b` from the means / covariances. Most of the linear algebra you'd need is already in `core/math.py`.
3. The existing `casteer` branch is the simplest reference; `midsteer` is the most involved.

### Testing changes

The repo has no formal test suite — verification is end-to-end through small smoke runs:

```bash
# Smoke test: tiny covariance + one strength, ~10 min on a single H100
NUM_COV_SAMPLES=500 STRENGTHS="2.0" \
    bash exp/sh/01_llm_concept_switching.sh
```

If `concept_scoring.py` and `consistency_scoring.py` produce non-empty `.tsv` files at the end, the change didn't break the data path. Compare absolute numbers against your last known-good run.

### Known gaps

- **AxBench/Gemma-2-2B evaluation** (rebuttal Tab. 1) is not in this repo. See the note in the "Reproducing paper results" section.
- **`I2P` evaluation** for diffusion nudity removal exists in code (`scripts/diffusion/run_i2p_eval.py`, `exp/datasets/eval/concepts/template_nudity.json`) but is not currently called from any example script.
- No automated tests. Adding `pytest` smoke tests around `core/math.py` and `core/pickle.py` would be a fast first contribution.

### Contributing

Pull requests welcome. Please keep PRs focused (one experiment / one abstraction at a time), and run a smoke test of `01_llm_concept_switching.sh` with `NUM_COV_SAMPLES=500` before opening.
