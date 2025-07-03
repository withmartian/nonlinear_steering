import openai
import csv
import time
import json


# Load prompts and responses from JSON
with open("prompts_and_responses.json", "r") as f:
    data = json.load(f)

results = []
column_name = "response"

for item in data:
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
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0,
        )

        scores = completion.choices[0].message['content']

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

    time.sleep(1.2)  # Respect rate limits

# Compute averages
valid_results = [r for r in results if r["text_quality"] and r["helpfulness"]]
avg_tq = sum(r["text_quality"] for r in valid_results) / len(valid_results)
avg_hp = sum(r["helpfulness"] for r in valid_results) / len(valid_results)

print(f"\nAverage Text Quality: {avg_tq:.2f}")
print(f"Average Helpfulness: {avg_hp:.2f}")

# Save to CSV
with open("evaluated_responses.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "response", "text_quality", "helpfulness"])
    writer.writeheader()
    writer.writerows(results)
