import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Replace this with your actual directory path
directory_path = "/home/ubuntu/nonlinear_steering/debates_multiple_layers_results"

# Data containers
success_by_alpha = defaultdict(list)
strength_by_alpha = defaultdict(list)

# Load files
for filename in os.listdir(directory_path):
    if filename.startswith("empirical_minus_strawman_steering_alpha_") and filename.endswith(".json"):
        with open(os.path.join(directory_path, filename), "r") as f:
            data = json.load(f)
            alpha = data["meta"]["alpha"]
            for step in data["step_runs"]:
                success_by_alpha[alpha].append((step["steps"], step["success_rate"]))
                strength_by_alpha[alpha].append((step["steps"], step["average_strength"]))

# Sort alphas
sorted_alphas = sorted(success_by_alpha.keys())

# Plot 1: Steering Success
plt.figure(figsize=(10, 6))
for alpha in sorted_alphas:
    steps, successes = zip(*sorted(success_by_alpha[alpha]))
    plt.plot(steps, successes, marker='o', label=f'α = {alpha}')
plt.title('Steering Success Rate vs. Steps')
plt.xlabel('Steps')
plt.ylabel('Success Rate')
plt.legend(title="Alpha")
plt.grid(True)
plt.tight_layout()
plt.savefig("steering_success.pdf")
plt.close()

# Plot 2: Steering Strength
plt.figure(figsize=(10, 6))
for alpha in sorted_alphas:
    steps, strengths = zip(*sorted(strength_by_alpha[alpha]))
    plt.plot(steps, strengths, marker='o', label=f'α = {alpha}')
plt.title('Steering Strength vs. Steps')
plt.xlabel('Steps')
plt.ylabel('Average Strength')
plt.legend(title="Alpha")
plt.grid(True)
plt.tight_layout()
plt.savefig("steering_strength.pdf")
plt.close()

print("Saved 'steering_success.pdf' and 'steering_strength.pdf'")

