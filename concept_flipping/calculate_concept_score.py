import json
import argparse

def calculate_average_score(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all scores
    scores = [item['score'] for item in data]
    
    # Calculate average
    average_score = sum(scores) / len(scores)
    
    print(f"Number of entries: {len(scores)}")
    print(f"Average score: {average_score:.2f}")
    
    return average_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing scores')
    args = parser.parse_args()
    
    calculate_average_score(args.file_path)
