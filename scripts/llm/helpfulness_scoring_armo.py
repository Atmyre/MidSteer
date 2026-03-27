"""
Score generated outputs for helpfulness using ArmoRM reward model.
Produces concept_scores_armo.tsv in the same format as concept_scores.tsv.

Usage:
    python scripts/llm/helpfulness_scoring_armo.py --dir <eval_dir>
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm

# Patch for ArmoRM compatibility with newer transformers
import transformers.models.llama.modeling_llama as _llama_module
if not hasattr(_llama_module, 'LLAMA_INPUTS_DOCSTRING'):
    _llama_module.LLAMA_INPUTS_DOCSTRING = ""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Directory with JSON output files')
    parser.add_argument('--model_name', type=str, default='RLHFlow/ArmoRM-Llama3-8B-v0.1')
    args = parser.parse_args()

    print(f"Loading ArmoRM ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    json_files = sorted(glob.glob(os.path.join(args.dir, '*.json')))
    if not json_files:
        print(f"No JSON files found in {args.dir}")
        return

    rows = []
    for file_path in tqdm.tqdm(json_files, desc="Scoring files"):
        fname = os.path.basename(file_path)

        with open(file_path) as f:
            data = json.load(f)

        scores = []
        for entry in data:
            prompt = entry.get('prompt', '')
            output = entry.get('output', '')
            if not output:
                continue

            # ArmoRM expects chat format
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_score = model(**inputs)
                score = output_score.score.item()
                scores.append(score)

        if scores:
            rows.append({
                'file': fname,
                'avg_score': np.mean(scores),
                'std': np.std(scores),
                'concept': 'helpfulness_armo',
                'num_samples': len(scores),
            })

    df = pd.DataFrame(rows)
    output_path = os.path.join(args.dir, 'concept_scores_armo.tsv')
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {output_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
