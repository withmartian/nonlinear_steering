from datasets import load_dataset
import json

# Load the dataset (default is 'alpaca_eval')
dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")

# Take the first 100 prompts
prompts = dataset[:100]['instruction']

prompts = [f"""Question:
{prompt}
Answer:
""" for prompt in prompts]

# Save to JSON file
with open("prompts_dump.json", "w") as f:
    json.dump(prompts, f, indent=2)
