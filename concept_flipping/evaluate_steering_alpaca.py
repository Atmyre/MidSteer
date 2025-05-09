import argparse
import json
import os
import glob
from typing import List, Dict
import torch
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def load_results(file_path: str) -> List[Dict]:
    """Load results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_bertscore(candidates: List[str], references: List[str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Calculate BERTScore between candidates and references."""
    P, R, F1 = score(candidates, references, lang='en', device=device)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
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
        'mean_score': sum(scores) / len(scores),
        'scores': scores
    }

def main(
    steered_path: str,
    unsteered_path: str,
    output_path: str,
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

    # Combine results
    evaluation_results = {
        'bertscore': bertscore_results,
        'bleu': bleu_results,
        'num_samples': len(steered_outputs)
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"BERTScore F1: {bertscore_results['f1']:.4f}")
    print(f"BERTScore Precision: {bertscore_results['precision']:.4f}")
    print(f"BERTScore Recall: {bertscore_results['recall']:.4f}")
    print(f"BLEU Score: {bleu_results['mean_score']:.4f}")

if __name__ == "__main__":

    # Get all steered files
    steered_files = glob.glob('concept_flipping/results/alpaca_instruct/*.json')

    for steered_file in steered_files:
        # Skip if this is the unsteered file
        if 'None' in steered_file:
            continue
            
        # Extract parameters from filename
        basename = os.path.basename(steered_file)
        # Reconstruct unsteered filename by replacing steering type with None
        if basename.startswith('casteer_'):
            unsteered_basename = 'None_' + basename.removeprefix('casteer_')
        elif basename.startswith('leace_'):
            unsteered_basename = 'None_' + basename.removeprefix('leace_')
        elif basename.startswith('mean_matching_'):
            unsteered_basename = 'None_' + basename.removeprefix('mean_matching_')
        else:
            raise ValueError(f"Unknown steering type {basename}")

        unsteered_file = os.path.join('concept_flipping/results/alpaca_instruct', unsteered_basename)
        
        if not os.path.exists(unsteered_file):
            print(f"Warning: Could not find unsteered file {unsteered_file}")
            continue
            
        # Create output path
        metrics_dir = 'concept_flipping/metrics'
        os.makedirs(metrics_dir, exist_ok=True)
        output_file = os.path.join(metrics_dir, basename.replace('.json', '_metrics.json'))
        
        print(f"\nEvaluating {basename}...")
        
        main(
            steered_path=steered_file,
            unsteered_path=unsteered_file, 
            output_path=output_file
        )
