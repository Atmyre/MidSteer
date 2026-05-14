"""Generate training questions for new concepts using GPT-4o-mini.

Produces the same format as existing training data:
  exp/datasets/train/{topic}_questions_batch_{i}.json

Each batch is a JSON list of ~100 question strings about the topic.
"""
import argparse
import json
import os

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are generating diverse questions about a specific topic for a research dataset. "
    "Generate exactly 100 unique, varied questions about the given topic. "
    "Questions should cover history, characteristics, cultural significance, practical aspects, "
    "comparisons, science, and everyday knowledge. "
    "Return ONLY a JSON array of strings, no other text."
)

USER_PROMPT_TEMPLATE = (
    "Generate 100 diverse questions about: {topic}\n\n"
    "The questions should be varied in style (how/what/why/when/where/can/do) "
    "and cover many different aspects of the topic. "
    "Return a JSON array of 100 question strings."
)


def generate_batch(client: OpenAI, topic: str, batch_idx: int, model: str) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
        ],
        temperature=1.0,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    # Handle both {"questions": [...]} and bare [...]
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return parsed


def main():
    parser = argparse.ArgumentParser(description='Generate training questions for new concepts')
    parser.add_argument('--topics', type=str, nargs='+', required=True,
                        help='Topics to generate questions for (e.g., chihuahuas muffins)')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches per topic')
    parser.add_argument('--output_dir', type=str, default='exp/datasets/train',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model')

    args = parser.parse_args()
    client = OpenAI()
    os.makedirs(args.output_dir, exist_ok=True)

    for topic in args.topics:
        print(f"Generating questions for: {topic}")
        for i in range(1, args.num_batches + 1):
            out_path = os.path.join(args.output_dir, f"{topic}_questions_batch_{i}.json")
            if os.path.exists(out_path):
                print(f"  Batch {i} already exists, skipping")
                continue

            print(f"  Batch {i}/{args.num_batches}...")
            questions = generate_batch(client, topic, i, args.model)
            with open(out_path, 'w') as f:
                json.dump(questions, f, indent=2)
            print(f"  Saved {len(questions)} questions to {out_path}")


if __name__ == "__main__":
    main()
