import argparse
import json
import os
import glob
from typing import List, Dict
import pandas as pd
import torch
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import numpy as np


def load_results(file_path: str) -> List[Dict]:
    """Load results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_bertscore(candidates: List[str], references: List[str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Calculate BERTScore between candidates and references."""
    P, R, F1 = score(candidates, references, lang='en', device=device)
    return {
        'bert_precision': P.mean().item(),
        'bert_recall': R.mean().item(),
        'bert_f1': F1.mean().item()
    }


def calculate_bleu(candidates: List[str], references: List[str]) -> Dict:
    """Calculate BLEU score between candidates and references."""
    smoothie = SmoothingFunction().method1
    scores = []
    
    for candidate, reference in zip(candidates, references):
        # Tokenize the texts
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        # Calculate BLEU score
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
        scores.append(score)
    
    return {
        'bleu_mean': np.mean(scores),
        'bleu_std': np.std(scores),
    }


def evaluate_file(
    steered_path: str,
    unsteered_path: str,
):
    # Load results
    steered_results = load_results(steered_path)
    unsteered_results = load_results(unsteered_path)

    # Extract outputs
    steered_outputs = [result['output'] for result in steered_results]
    unsteered_outputs = [result['output'] for result in unsteered_results]

    # Calculate metrics
    print("Calculating BERTScore...")
    bertscore_results = calculate_bertscore(steered_outputs, unsteered_outputs)
    
    print("Calculating BLEU score...")
    bleu_results = calculate_bleu(steered_outputs, unsteered_outputs)

    return {
        'file': os.path.basename(steered_path),
        **bertscore_results,
        **bleu_results,
        'num_samples': len(steered_outputs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()


    unsteered_file = os.path.join(args.dir, 'None_0.0.json')
    if not os.path.exists(unsteered_file):
        raise FileNotFoundError(f"Unsteered file not found at {unsteered_file}")
    
    files = glob.glob(f'{args.dir}/*.json')

    data = []

    for file in files:
        if file == unsteered_file:
            continue

        print(f"Evaluating {file}...")
        results = evaluate_file(
            steered_path=file,
            unsteered_path=unsteered_file,
        )
        data.append(results)

    df = pd.DataFrame(data)
    df.to_csv(f'{args.dir}/consistency_scores.tsv', index=False, sep='\t', encoding='utf-8')
