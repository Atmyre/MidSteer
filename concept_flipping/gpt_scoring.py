import os
import json
import argparse
import time
from openai import OpenAI
from typing import Dict, List
import tqdm
from pydantic import BaseModel

class Response(BaseModel):
    score: float

def score_text(client: OpenAI, text: str, concept: str) -> float:
    """Score a text from 0-10 on relevance to concept using GPT."""
    try:
        prompt = (
            f"Please score the text on a scale of 0 to 10, where the score reflects the extent "
            f"to which it demonstrates relevance to the concept of {concept}. A score of 0 "
            f"indicates the text is completely irrelevant and shows no connection to {concept}, "
            f"whereas a score of 10 signifies strong and explicit relevance to {concept}. For "
            f"texts that suggest varying degrees of relevance, use the intermediate numbers "
            f"(1 through 9) to represent the gradation of connection to {concept}. Output the result as json with a single 'score' field.\n\n"
            f"Text to evaluate: ```"
            f"{text}"
            f"```"
        )
        
        response = client.responses.parse(
            model="o4-mini",
            input=[{"role": "user", "content": prompt}],
            text_format=Response,
        )
        
        score = response.output_parsed.score
        return min(max(score, 0), 10)  # Ensure score is between 0-10
        
    except Exception as e:
        print(f"Error scoring text: {e}")
        return -1

def process_file(client: OpenAI, concept: str, file_path: str) -> List[Dict]:
    """Process a single JSON file and score its content."""
    results = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for entry in tqdm.tqdm(data):
            if 'output' not in entry:
                print(f"Warning: Entry missing 'output' field, skipping")
                continue
                
            score = score_text(client, entry['output'], concept)
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
    parser.add_argument('concept', type=str, help='Concept to score against')
    parser.add_argument('filename', type=str, help='JSON file to process')
    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI()  # Assumes OPENAI_API_KEY is set in environment variables
    
    # Process file and get scores
    results = process_file(client, args.concept, args.filename)
    
    # Calculate and display average score
    print(f"\nScoring results for concept: {args.concept}")
    print("-" * 50)
    
    if results:
        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores)
        print(f"Average score = {avg_score:.2f}")
        
        # Save results to file
        scores_dir = os.path.join('concept_flipping', 'scores')
        os.makedirs(scores_dir, exist_ok=True)
        output_file = os.path.join(scores_dir, f"scores_{args.concept}_{os.path.basename(args.filename)}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")
    else:
        print("No valid scores were generated.")

if __name__ == "__main__":
    main()
