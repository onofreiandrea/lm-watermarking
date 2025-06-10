import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

with open('watermark_paraphrase_results_with_temp_0.3.json') as f:
    data = json.load(f)
    
    df = pd.DataFrame(data)

    # Get combined min and max for x-axis from both columns
    x_min = min(df['sequence_prob_score'].min(), df['paraphrased_sequence_prob_score'].min())
    x_max = max(df['sequence_prob_score'].max(), df['paraphrased_sequence_prob_score'].max())

    # Define bins once for both histograms using combined range
    bins = np.linspace(x_min, x_max, 16)  # 15 bins

    # Compute histograms to find max y-axis value for both
    hist1, _ = np.histogram(df['sequence_prob_score'], bins=bins)
    hist2, _ = np.histogram(df['paraphrased_sequence_prob_score'], bins=bins)
    y_max = max(hist1.max(), hist2.max()) + 1  # add some padding

    plt.figure(figsize=(6, 8))  # taller figure

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
    plt.hist(df['sequence_prob_score'], bins=bins, color='blue', alpha=0.7)
    plt.title('Histogram of sequence_prob_score')
    plt.xlabel('sequence_prob_score')
    plt.ylabel('Frequency')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
    plt.hist(df['paraphrased_sequence_prob_score'], bins=bins, color='green', alpha=0.7)
    plt.title('Histogram of paraphrased_sequence_prob_score')
    plt.xlabel('paraphrased_sequence_prob_score')
    plt.ylabel('Frequency')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)

    plt.tight_layout()
    plt.show()