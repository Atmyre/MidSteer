"""
Evaluate steering methods on RealToxicityPrompts (Gehman et al., 2020).
Uses raw text completion (no chat template) so the model generates continuations
rather than refusing. Scores outputs with detoxify.

Usage:
    python scripts/llm/run_rtp_eval.py \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --layer_type self_attn \
        --source_concept_path steering_vectors/toxicity.pt \
        --target_concept_path steering_vectors/helpfulness.pt \
        --steer_type midsteer --strength 3.0 \
        --mu_neutral covariances/means.pt \
        --cov_neutral covariances/covariances.pt \
        --output_dir results/rtp/midsteer_3.0 \
        --num_prompts 500 --min_toxicity 0.5
"""
import argparse
import json
import os
from contextlib import contextmanager

import torch
import tqdm
from datasets import load_dataset
from detoxify import Detoxify
from transformers import GenerationConfig

from core.controller import CrossAttentionOutputSteering, ModelToSteer, SteeringVectors
from core.llm_steering import llm_register_vector_control
from core.pickle import unpickle
from core.utils import init_llm_model_and_tokenizer, get_device


def main(args):
    device = get_device()

    # Load model
    print("Loading model...")
    model, tokenizer = init_llm_model_and_tokenizer(model_name=args.model_name)

    # Set up steering if requested
    if args.steer_type and args.source_concept_path:
        source_concept = unpickle(args.source_concept_path)
        target_concept = unpickle(args.target_concept_path)
        mu_neutral = unpickle(args.mu_neutral) if args.mu_neutral else None
        cov_neutral = unpickle(args.cov_neutral) if args.cov_neutral else None

        control = CrossAttentionOutputSteering(
            model_to_steer=ModelToSteer.LLAMA,
            steer_type=args.steer_type,
            steer_back=True,
            device=device,
            source_concepts=[source_concept],
            target_concepts=[target_concept],
            mu_neutral=mu_neutral,
            sigma_neutral=cov_neutral,
            strength=args.strength,
        )
        label = f"{args.steer_type}_{args.strength}"
    else:
        control = None
        label = "baseline"

    # Load RealToxicityPrompts
    print("Loading RealToxicityPrompts...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Filter by toxicity threshold
    prompts = []
    for ex in ds:
        tox = ex["prompt"]["toxicity"]
        if tox is not None and tox >= args.min_toxicity:
            prompts.append({
                "text": ex["prompt"]["text"],
                "toxicity": tox,
            })
    prompts.sort(key=lambda x: -x["toxicity"])
    prompts = prompts[:args.num_prompts]
    print(f"Selected {len(prompts)} prompts with toxicity >= {args.min_toxicity}")

    # Generate completions
    os.makedirs(args.output_dir, exist_ok=True)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    @contextmanager
    def dummy_context():
        yield

    results = []
    for p in tqdm.tqdm(prompts, desc=f"Generating ({label})"):
        # Raw tokenization - no chat template
        inputs = tokenizer(p["text"], return_tensors="pt").input_ids.to(device)

        if control is not None:
            ctx = llm_register_vector_control(
                model=model,
                control=[control],
                layer_type=args.layer_type,
                min_token_index=inputs.shape[1],
            )
        else:
            ctx = dummy_context()

        with ctx, torch.no_grad():
            output_ids = model.generate(inputs, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id)

        # Decode only the generated part
        generated = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

        results.append({
            "prompt": p["text"],
            "prompt_toxicity": p["toxicity"],
            "output": generated,
        })

    # Save raw outputs
    output_file = os.path.join(args.output_dir, f"{label}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Score with detoxify
    print("Scoring with detoxify...")
    det_model = Detoxify("original", device="cuda")
    texts = [r["output"] for r in results]

    batch_size = 32
    all_tox = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        scores = det_model.predict(batch)
        all_tox.extend(scores["toxicity"])

    for r, t in zip(results, all_tox):
        r["output_toxicity"] = t

    # Save scored outputs
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    import numpy as np
    avg_tox = np.mean(all_tox)
    std_tox = np.std(all_tox)
    print(f"\n{label}: avg_toxicity={avg_tox:.6f}, std={std_tox:.6f}, n={len(all_tox)}")

    # Save summary
    summary_file = os.path.join(args.output_dir, "rtp_scores.tsv")
    write_header = not os.path.exists(summary_file)
    with open(summary_file, "a") as f:
        if write_header:
            f.write("label\tavg_toxicity\tstd_toxicity\tnum_samples\n")
        f.write(f"{label}\t{avg_tox:.6f}\t{std_tox:.6f}\t{len(all_tox)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer_type", type=str, default="self_attn")
    parser.add_argument("--source_concept_path", type=str, default=None)
    parser.add_argument("--target_concept_path", type=str, default=None)
    parser.add_argument("--steer_type", type=str, default=None, choices=["casteer", "leace", "midsteer"])
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--mu_neutral", type=str, default=None)
    parser.add_argument("--cov_neutral", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_prompts", type=int, default=500)
    parser.add_argument("--min_toxicity", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()
    main(args)
