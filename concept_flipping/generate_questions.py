import openai
import json
import time
from typing import List
import os
from pathlib import Path
from pprint import pprint
import argparse

# Initialize OpenAI client
client = openai.OpenAI()

def generate_questions_batch(topic: str, model: str, num_questions: int = 100) -> List[str]:
    """
    Generate a batch of questions about a specific topic using OpenAI API.
    
    Args:
        topic: The topic to generate questions about
        model: The model to use for generation
        num_questions: Number of questions to generate in this batch
        
    Returns:
        List of generated questions
    """
    prompt = f"""Generate {num_questions} diverse questions about {topic}. 
    The questions should be:
    - Concise but allow for detailed answers
    - In English
    - Cover various aspects of {topic} (history, mechanics, care, usage, etc.)
    - Suitable for testing concept understanding in language models
    - Each question should be on a new line
    - Do not include numbers or bullet points
    - Do not include any explanations or additional text
    
    Just provide the questions, one per line."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse, high-quality questions."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=64_000
        )
        
        # Extract questions from response
        questions = response.choices[0].message.content.strip().split('\n')
        # pprint(response)
        # Clean up any empty lines or formatting artifacts
        questions = [q.strip() for q in questions if q.strip()]
        if not questions:
            raise ValueError("Generated questions are empty")
        
        return questions[:num_questions]  # Ensure we don't return more than requested
    
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def save_questions(questions: List[str], topic: str, batch_num: int):
    """Save generated questions to a JSON file."""
    output_dir = Path("concept_flipping/questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"{topic}_questions_batch_{batch_num}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(questions)} questions to {filename}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate questions about specified topics using OpenAI API.')
    parser.add_argument('--topics', nargs='+', required=True,
                      help='List of topics to generate questions about')
    parser.add_argument('--model', default='o4-mini',
                      help='Model to use for generation (default: o4-mini)')
    parser.add_argument('--total-questions', type=int, default=1000,
                      help='Total number of questions to generate per topic (default: 1000)')
    parser.add_argument('--questions-per-batch', type=int, default=100,
                      help='Number of questions to generate per batch (default: 100)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    for topic in args.topics:
        print(f"\nGenerating questions about {topic}...")
        
        batches_per_topic = args.total_questions // args.questions_per_batch
        
        for batch in range(batches_per_topic):
            print(f"Generating batch {batch + 1}/{batches_per_topic}...")
            
            questions = generate_questions_batch(topic, args.model, args.questions_per_batch)
            if questions:
                save_questions(questions, topic, batch + 1)
            
            # Add a small delay between batches to avoid rate limiting
            time.sleep(2)

if __name__ == "__main__":
    main()
