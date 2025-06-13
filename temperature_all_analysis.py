import json
import matplotlib.pyplot as plt

# Define your files and labels
files = [
    ('watermark_paraphrase_results_with_temp_1_0.3.json', '1 0.3'),
    ('watermark_paraphrase_results_with_temp_t_0.3.json', 't 0.3'),
    ('watermark_paraphrase_results_with_temp_1_0.5.json', '1 0.5'),
    ('watermark_paraphrase_results_with_temp_t_0.5.json', 't 0.5')
]

# First pass: compute global y-axis limits
all_differences = []

for filename, _ in files:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    diffs = [
        entry['sequence_prob_score'] - entry['paraphrased_sequence_prob_score']
        for entry in data
        if abs(entry['sequence_prob_score'] - entry['paraphrased_sequence_prob_score']) <= 0.5
    ]
    all_differences.extend(diffs)

global_min = min(all_differences)
global_max = max(all_differences)

# Set up 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
axs = axs.flatten()

# Second pass: plot
for i, (filename, label) in enumerate(files):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Compute and clean differences
    differences = [
        entry['sequence_prob_score'] - entry['paraphrased_sequence_prob_score']
        for entry in data
        if abs(entry['sequence_prob_score'] - entry['paraphrased_sequence_prob_score']) <= 0.5
    ]
    
    differences.sort()
    
    avg = sum(differences) / len(differences)
    below_avg = [d for d in differences if d < avg]
    above_avg = [d for d in differences if d > avg]
    
    avg_below = sum(below_avg) / len(below_avg) if below_avg else 0
    avg_above = sum(above_avg) / len(above_avg) if above_avg else 0

    # Print stats
    print(f"\n{label}:")
    print(f"  Total points: {len(differences)}")
    print(f"  Overall average: {avg:.4f}")
    print(f"  Average below avg: {avg_below:.4f} ({len(below_avg)} points)")
    print(f"  Average above avg: {avg_above:.4f} ({len(above_avg)} points)")

    # Plot
    axs[i].plot(differences, marker='o', linestyle='-', markersize=3, label='Sorted Difference')
    axs[i].axhline(avg, color='red', linestyle='--', label=f'Avg = {avg:.4f}')
    axs[i].axhline(avg_below, color='blue', linestyle=':', label=f'Avg < avg = {avg_below:.4f}')
    axs[i].axhline(avg_above, color='green', linestyle=':', label=f'Avg > avg = {avg_above:.4f}')
    
    axs[i].set_ylim(global_min, global_max)
    axs[i].set_title(f"Sorted Score Differences ({label})")
    axs[i].set_xlabel("Sorted Index")
    axs[i].set_ylabel("Score Difference")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
