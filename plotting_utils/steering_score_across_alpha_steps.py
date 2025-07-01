import os
import json
import pandas as pd

def compute_best_steering_scores(directory):
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

            # Extract target and avoid styles from the metadata
            target = data["meta"]["target_style"]
            avoid = data["meta"]["avoid_style"]
            combination = f"{target}_vs_{avoid}"

            best_score = -1
            best_alpha = None
            best_step = None

            for alpha_entry in data.get("alphas", []):
                alpha = alpha_entry["alpha"]
                for step_run in alpha_entry.get("step_runs", []):
                    steps = step_run["steps"]
                    success_rate = step_run["success_rate"]
                    strength = step_run["average_strength"]
                    score = (strength / 5.0) * success_rate  # Normalized steering score

                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_step = steps

            results.append({
                "Combination": combination,
                "Best Alpha": best_alpha,
                "Best Step": best_step,
                "Best Steering Score": round(best_score, 4)
            })

    # Create a summary table as a pandas DataFrame
    return pd.DataFrame(results)
def save_as_latex_table(df, output_path):
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Combination & Best Alpha & Best Step & Best Steering Score \\\\")
    latex_lines.append("\\midrule")

    for _, row in df.iterrows():
        latex_lines.append(f"{row['Combination']} & {row['Best Alpha']} & {row['Best Step']} & {row['Best Steering Score']} \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

df = compute_best_steering_scores("/home/ubuntu/nonlinear_steering/notebooks/linear_steering_results")
save_as_latex_table(df, "linear_steering_best_alpha_step_steering.tex")
