import openai
import csv
import time
import json
import pandas as pd

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

# filename = "alpaca_llama-3.2-3b_1_empathetic.csv"
# filename = "alpaca_llama-3.2-3b_2_casual-empathetic.csv"
# filename = "alpaca_llama-3.2-3b_3_cautious-concise-empathetic.csv"

# filename = "alpaca_mistral-7b_1_expert.csv"
filename = "alpaca_mistral-7b_2_casual-cautious.csv"
# filename = "alpaca_mistral-7b_2_casual-empathetic.csv"

# filename = "alpaca_olmo-2-7b_1_empathetic.csv"
# filename = "alpaca_olmo-2-7b_2_casual-concise.csv"
# filename = "alpaca_olmo-2-7b_3_casual-cautious-concise.csv"

# Load prompts and responses from pandas

df = pd.read_csv(filename)

data = df.to_dict(orient="records")[:100]

results = []
# column_name = "caa"
# column_name = "k_steering"
column_name = "unsteered"

for item in tqdm(data):
    prompt = item["prompt"]
    response = item[column_name]

    # Ask GPT-4 to evaluate
    evaluation_prompt = f"""
You are an expert evaluator. Given the following prompt and response, rate the **text quality** and **helpfulness** of the response.

**Prompt:** {prompt}

**Response:** {response}

Rate on a scale of 1 to 10 (higher is better):

- Text Quality (fluency, structure, grammar)
- Helpfulness (how well it answers the prompt)

Return JSON like: {{ "text_quality": x, "helpfulness": y }}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0,
        )

        scores = str(completion.choices[0].message.content).strip()

        # Attempt to parse JSON response
        scores_dict = json.loads(scores)
        tq = scores_dict.get("text_quality", None)
        hp = scores_dict.get("helpfulness", None)

        results.append({
            "prompt": prompt,
            "response": response,
            "text_quality": tq,
            "helpfulness": hp
        })

    except Exception as e:
        print(f"Error processing item: {e}")
        continue


# Compute averages
valid_results = [r for r in results if r["text_quality"] and r["helpfulness"]]
avg_tq = sum(r["text_quality"] for r in valid_results) / len(valid_results)
avg_hp = sum(r["helpfulness"] for r in valid_results) / len(valid_results)

print(f"\nAverage Text Quality: {avg_tq:.2f}")
print(f"Average Helpfulness: {avg_hp:.2f}")
print(f"Column type was {column_name} for {filename}")

# Save to CSV
with open(f"{filename}_evaluated_responses.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "response", "text_quality", "helpfulness"])
    writer.writeheader()
    writer.writerows(results)
