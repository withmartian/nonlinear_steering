import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.cm import get_cmap

# Load and group data
directory_path = "/home/ubuntu/nonlinear_steering/results_all_layers_steering/results_across_steps_alphas"
score_by_alpha = defaultdict(list)

for filename in os.listdir(directory_path):
    if filename.startswith("empirical_minus_strawman_steering_alpha_") and filename.endswith(".json"):
        with open(os.path.join(directory_path, filename), "r") as f:
            data = json.load(f)
            alpha = data["meta"]["alpha"]
            for step in data["step_runs"]:
                steps = step["steps"]
                success = step["success_rate"]
                strength = step["average_strength"]
                score = (strength / 5.0) * success
                score_by_alpha[alpha].append((steps, score))

early_alphas = sorted([a for a in score_by_alpha if a <= 1.0])
mid_alphas   = sorted([a for a in score_by_alpha if 1.0 <= a <= 3.0])
late_alphas  = sorted([a for a in score_by_alpha if a > 3.0])

def get_shaded_colors(cmap_name, n_colors, min_val=0.3, max_val=0.9):
    cmap = get_cmap(cmap_name)
    return [cmap(min_val + (max_val - min_val) * i / max(n_colors - 1, 1)) for i in range(n_colors)]

# Create 2x2 grid
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# Top row
ax1 = fig.add_subplot(gs[0, 0])  # Early
ax2 = fig.add_subplot(gs[0, 1])  # Mid

# Bottom row center by adding inset axes
bottom_ax_rect = [0.33, 0.1, 0.34, 0.35]  # [left, bottom, width, height] (manual placement)
ax3 = fig.add_axes(bottom_ax_rect)       # Late

configs = [
    (early_alphas, "Blues", "Early Alphas", ax1),
    (mid_alphas, "Greens", "Middle Alphas", ax2),
    (late_alphas, "Reds", "Late Alphas", ax3),
]

for alphas, cmap, title, ax in configs:
    colors = get_shaded_colors(cmap, len(alphas))
    for alpha, color in zip(alphas, colors):
        steps, scores = zip(*sorted(score_by_alpha[alpha]))
        ax.plot(steps, scores, marker='o', label=f'α = {alpha}', color=color)
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.grid(True)
    ax.legend(title="Alpha", fontsize=8)
    if ax == ax1:
        ax.set_ylabel('Normalized Steering Score')

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for the manually placed ax3
plt.savefig("steering_score_3col_centered.pdf")
plt.close()
print("Saved 'steering_score_3col_centered.pdf'")













