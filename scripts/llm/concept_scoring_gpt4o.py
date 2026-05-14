import os
import glob
import json
import argparse
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm
from openai import OpenAI, RateLimitError, APIError


PROMPT_TEMPLATE = (
    "Score the text on a scale of 0 to 10, where the score reflects the extent "
    "to which it demonstrates relevance to the concept of {concept}. A score of 0 "
    "indicates the text is completely irrelevant and shows no connection to {concept}, "
    "whereas a score of 10 signifies strong and explicit relevance to {concept}. For "
    "texts that suggest varying degrees of relevance, use the intermediate numbers "
    "(1 through 9) to represent the gradation of connection to {concept}. Output just a single number.\n\n"
    "Text to evaluate: ```{text}```"
)


def score_text(client: OpenAI, text: str, concept: str, model: str) -> float:
    prompt = PROMPT_TEMPLATE.format(concept=concept, text=text)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            score = int(response.choices[0].message.content.strip())
            if 0 <= score <= 10:
                return score
            return -1
        except RateLimitError:
            time.sleep(2 ** attempt)
        except (ValueError, IndexError):
            return -1
        except APIError:
            return -1
    return -1


def score_entry(client: OpenAI, text: str, concept: str, model: str, prompt_text: str, file: str):
    """Score a single entry — designed for use with ThreadPoolExecutor."""
    score = score_text(client, text, concept, model)
    return {
        'file': file,
        'concept': concept,
        'prompt': prompt_text,
        'output': text,
        'score': score,
    }


def main():
    parser = argparse.ArgumentParser(description='Score text relevance to a concept using GPT-4o-mini')
    parser.add_argument('--concept', type=str, nargs='+', required=True, help='Concept(s) to score against')
    parser.add_argument('--dir', type=str, required=True, help='Directory with JSON files to score')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--output_suffix', type=str, default='_gpt4o',
                        help='Suffix for output files (default: _gpt4o)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of concurrent API workers (default: 32)')

    args = parser.parse_args()
    client = OpenAI()

    files = sorted(glob.glob('**/*.json', recursive=True, root_dir=args.dir))
    if not files:
        print(f"No JSON files found in {args.dir}")
        return

    # Build all tasks upfront
    tasks = []
    for concept in set(args.concept):
        for file in files:
            file_path = os.path.join(args.dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for entry in data:
                    if 'output' in entry:
                        tasks.append((entry['output'], concept, entry.get('prompt', ''), file))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Scoring {len(tasks)} entries with {args.workers} workers...")

    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_entry, client, text, concept, args.model, prompt_text, file): idx
            for idx, (text, concept, prompt_text, file) in enumerate(tasks)
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            result = future.result()
            if result['score'] >= 0:
                all_results.append(result)

    # Summary TSV (same format as concept_scoring.py)
    summary_data = {'file': [], 'avg_score': [], 'std': [], 'concept': []}
    df_all = pd.DataFrame(all_results)
    for (file, concept), group in df_all.groupby(['file', 'concept']):
        summary_data['file'].append(file)
        summary_data['avg_score'].append(group['score'].mean())
        summary_data['std'].append(group['score'].std())
        summary_data['concept'].append(concept)

    df = pd.DataFrame(summary_data)
    out_path = os.path.join(args.dir, f'concept_scores{args.output_suffix}.tsv')
    df.to_csv(out_path, index=False, sep='\t', encoding='utf-8')
    print(f"Summary scores saved to {out_path}")

    # Per-sample scores
    df_samples = df_all[['file', 'concept', 'prompt', 'score']]
    per_sample_path = os.path.join(args.dir, f'concept_scores{args.output_suffix}_per_sample.tsv')
    df_samples.to_csv(per_sample_path, index=False, sep='\t', encoding='utf-8')
    print(f"Per-sample scores saved to {per_sample_path}")


if __name__ == "__main__":
    main()
