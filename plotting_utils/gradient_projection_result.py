import os
import re
import pandas as pd

def extract_scores_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    combinations = content.split("### Selected Style: Avoid = ")[1:]
    data = []

    for block in combinations:
        match_avoid = re.match(r"([a-zA-Z_]+) ###", block)
        avoid_style = match_avoid.group(1) if match_avoid else "unknown"

        caa_score_match = re.search(r"'caa_steering_score': ([0-9.]+)", block)
        k_score_match = re.search(r"'k_steer_steering_score': ([0-9.]+)", block)

        caa_score = float(caa_score_match.group(1)) if caa_score_match else None
        k_score = float(k_score_match.group(1)) if k_score_match else None

        data.append({
            "Avoid Style": avoid_style,
            "CAA Steering Score": caa_score,
            "K-Steer Steering Score": k_score,
        })

    return pd.DataFrame(data)

def generate_latex_table_from_folder(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            df = extract_scores_from_file(full_path)
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        latex_table = final_df.to_latex(index=False, float_format="%.3f",
                                        caption="Steering scores for CAA and K-Steer across different avoid styles.",
                                        label="tab:steering_scores")
        output_path = os.path.join(folder_path, "steering_scores_table.tex")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        return final_df, output_path
    else:
        return pd.DataFrame(), None

df, latex_path = generate_latex_table_from_folder("/home/ubuntu/nonlinear_steering/results_all_layers_steering/gradient_projection")
print(df)
print(f"LaTeX saved to: {latex_path}")