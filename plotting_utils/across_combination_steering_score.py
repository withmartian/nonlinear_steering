import os
import json
import pandas as pd

# Path to your folder
directory_path = "/home/ubuntu/nonlinear_steering/results_all_layers_steering/across_combinations_caa"

# Function to compute steering score
def compute_steering_score(success, strength):
    return (strength / 5.0) * success

# Collect detailed records
all_results = []

# Scan directory
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            target = data["meta"].get("target_style")
            avoid = data["meta"].get("avoid_style")
            steps = data["meta"].get("steps", None)

            for step_entry in data["step_runs"]:
                alpha = step_entry["alpha"]
                success = step_entry["success_rate"]
                strength = step_entry["average_strength"]
                score = compute_steering_score(success, strength)

                all_results.append({
                    "combination": f"{target}_vs_{avoid}",
                    "alpha": alpha,
                    "steps": steps,
                    "success_rate": success,
                    "strength": strength,
                    "steering_score": score
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to DataFrame (file column is already excluded)
df_all = pd.DataFrame(all_results)

# Save full table
df_all.to_csv("caa_lookup_alpha_step_per_combination.csv", index=False)

# Find best alpha/step pair per combination
df_best = (
    df_all.loc[df_all.groupby("combination")["steering_score"].idxmax()]
    .sort_values(by="combination")
    .reset_index(drop=True)
)

# Save best score table
df_best.to_csv("caa_best_alpha_step_per_combination.csv", index=False)

# Display
print("All Steering Scores Table:")
print(df_all.head())

print("\nBest Alpha and Step per Combination:")
print(df_best)
