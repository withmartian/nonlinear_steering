import json

def extract_scores(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for combo in data["combinations"]:
        avoid_style = combo["avoid_style"]
        caa_score = combo.get("caa_steering_score", 0.0)
        k_score = combo.get("k_steer_steering_score", 0.0)
        rows.append((avoid_style, caa_score, k_score))
    
    # Sort alphabetically by avoid style for consistency
    rows.sort()

    return rows

def generate_latex_table(rows):
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Avoid Style} & \\textbf{CAA} & \\textbf{K-Steering} \\\\")
    latex.append("\\midrule")
    
    for avoid_style, caa_score, k_score in rows:
        bold_caa = f"\\textbf{{{caa_score:.3f}}}" if caa_score > k_score else f"{caa_score:.3f}"
        bold_k = f"\\textbf{{{k_score:.3f}}}" if k_score > caa_score else f"{k_score:.3f}"
        latex.append(f"{avoid_style} & {bold_caa} & {bold_k} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Steering scores for Projection Removal (K-Steering Algorithm 2) versus Directional Ablation (CAA) across different avoid styles.}")
    latex.append("\\end{table}")
    return "\n".join(latex)

# Example usage
rows = extract_scores("/home/ubuntu/nonlinear_steering/results_all_layers_steering/gradient_projection/gradient_projection_results.json")  # Replace with your JSON file path
latex_code = generate_latex_table(rows)

# Save to .tex
with open("steering_scores_table.tex", "w") as f:
    f.write(latex_code)

print(latex_code)
