from datasets import load_dataset
import json

# Load the dataset (default is 'alpaca_eval')
dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")

# Take the first 200 prompts
prompts = dataset[:200]['instruction']


# Save to JSON file
with open("prompts_dump.json", "w") as f:
    json.dump(prompts, f, indent=2)
