import os
import glob
import numpy as np
import json
import argparse
from typing import Dict, List
import pandas as pd
import torch
import tqdm
from pydantic import BaseModel


from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ALPACA_DEFAULT_INSTRUCTION, tokenize_with_chat_template
from utils import init_llm_model_and_tokenizer

class Response(BaseModel):
    score: float

def score_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, concept: str, digits_token_ids: list[int]) -> float:
    """Score a text from 0-10 relevance to concept using GPT."""
    try:
        prompt = (
            f"Score the text on a scale of 0 to 10, where the score reflects the extent "
            f"to which it demonstrates relevance to the concept of {concept}. A score of 0 "
            f"indicates the text is completely irrelevant and shows no connection to {concept}, "
            f"whereas a score of 10 signifies strong and explicit relevance to {concept}. For "
            f"texts that suggest varying degrees of relevance, use the intermediate numbers "
            f"(1 through 9) to represent the gradation of connection to {concept}. Output just a single number.\n\n"
            f"Text to evaluate: ```"
            f"{text}"
            f"```"
        )


        with torch.no_grad():
            inputs = tokenize_with_chat_template(
                tokenizer,
                system_prompt=ALPACA_DEFAULT_INSTRUCTION,
                user_input=prompt,
            )
            response = model.forward(inputs.to(model.device))
            logits_slice = response.logits[0, -1, digits_token_ids]
            ret = torch.argmax(logits_slice).cpu().detach().numpy()
            
        
        return ret
        
    except Exception as e:
        print(f"Error scoring text: {e}")
        return -1

def process_file(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, concept: str, file_path: str) -> List[Dict]:
    """Process a single JSON file and score its content."""
    results = []

    digits_token_ids = []
    for i in range(11):
        digits_token_ids.append(tokenizer.convert_tokens_to_ids(f'{i}'))

    # print(digits_token_ids)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for entry in tqdm.tqdm(data):
            if 'output' not in entry:
                print(f"Warning: Entry missing 'output' field, skipping")
                continue
                
            score = score_text(model, tokenizer, entry['output'], concept, digits_token_ids)
            if score >= 0:  # Only include valid scores
                result = {
                    'prompt': entry.get('prompt', ''),
                    'output': entry['output'],
                    'score': score
                }
                results.append(result)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Score text relevance to a concept using GPT')
    parser.add_argument('--concept', type=str, nargs='+', help='Concept to score against')
    parser.add_argument('--dir', type=str, help='Subdirectory to process')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    
    args = parser.parse_args()
    model, tokenizer = init_llm_model_and_tokenizer(model_name=args.model_name)

    files = glob.glob(f'**/*.json', recursive=True, root_dir=args.dir)

    data = {
        'file': [],
        'avg_score': [],
        'std': [],
        'concept': [],
    }

    for concept in set(args.concept):
        for file in files:
            file_path = os.path.join(args.dir, file)
            results = process_file(model, tokenizer, concept, file_path)
            scores = [r['score'] for r in results]

            data['file'].append(file)
            data['avg_score'].append(np.mean(scores))
            data['std'].append(np.std(scores))
            data['concept'].append(concept)

    df = pd.DataFrame(data)
    df.to_csv(f'{args.dir}/scores.tsv', index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    main()
