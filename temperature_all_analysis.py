import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of your files and their parameters
files = [
    ('watermark_paraphrase_results_with_temp_1_0.3.json', '1 0.3'),
    ('watermark_paraphrase_results_with_temp_t_0.3.json', 't 0.3'),  # replace with your actual files
    ('watermark_paraphrase_results_with_temp_1_0.5.json', '1 0.5'),  # replace with your actual files
    ('watermark_paraphrase_results_with_temp_t_0.5.json', 't 0.5')   # replace with your actual files
]

# First pass to determine global min/max for consistent axes
all_original_scores = []
all_paraphrased_scores = []

for file, _ in files:
    with open(file) as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        all_original_scores.extend(df['sequence_prob_score'].tolist())
        all_paraphrased_scores.extend(df['paraphrased_sequence_prob_score'].tolist())

# Get global min and max for x-axis
x_min = min(min(all_original_scores), min(all_paraphrased_scores))
x_max = max(max(all_original_scores), max(all_paraphrased_scores))
bins = np.linspace(x_min, x_max, 16)  # 15 bins

# Get global max for y-axis by computing all histograms
max_counts = 0
for file, _ in files:
    with open(file) as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        hist1, _ = np.histogram(df['sequence_prob_score'], bins=bins)
        hist2, _ = np.histogram(df['paraphrased_sequence_prob_score'], bins=bins)
        current_max = max(hist1.max(), hist2.max())
        if current_max > max_counts:
            max_counts = current_max
y_max = max_counts + 1  # add some padding

# Create the figure with 4 rows and 2 columns
plt.figure(figsize=(12, 16))  # adjust size as needed

for i, (file, params) in enumerate(files):
    with open(file) as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        
        # Original scores plot
        plt.subplot(4, 2, 2*i+1)
        plt.hist(df['sequence_prob_score'], bins=bins, color='blue', alpha=0.7)
        plt.title(f'Original Scores (params: {params})')
        plt.ylabel('Frequency')
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        if i == len(files) - 1:  # only add xlabel to bottom plots
            plt.xlabel('sequence_prob_score')
        
        # Paraphrased scores plot
        plt.subplot(4, 2, 2*i+2)
        plt.hist(df['paraphrased_sequence_prob_score'], bins=bins, color='green', alpha=0.7)
        plt.title(f'Paraphrased Scores (params: {params})')
        plt.ylabel('Frequency')
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        if i == len(files) - 1:  # only add xlabel to bottom plots
            plt.xlabel('paraphrased_sequence_prob_score')

plt.tight_layout()
plt.show()