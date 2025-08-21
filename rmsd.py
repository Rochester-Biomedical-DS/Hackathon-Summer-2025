import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_rel
import sys

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def compare_csv(file):
    """
    To use this, in your terminal run this file with the name of the CSV file you want to compare as an argument.
    """
    data_reference = pd.read_csv('prediction/predictions_day1.csv')
    data_to_compare = pd.read_csv(file)

    merged_data = pd.merge(data_reference, data_to_compare, on=[
                           'gene', 'perturbation'], suffixes=('_ref', '_comp'))
    if len(merged_data) < len(data_reference):
        print("Warning: Some entries in the reference data are missing in the comparison data.")
    reference = merged_data['expression_ref']
    comparison = merged_data['expression_comp']

    diff = comparison - reference
    mad = np.mean(np.abs(diff))
    rmsd = np.sqrt(np.mean(diff**2))
    pearson_corr, _ = pearsonr(reference, comparison)
    spearman_corr, _ = spearmanr(reference, comparison)
    ttest_result = ttest_rel(reference, comparison)

    print("Comparison Results:")
    print(f"Mean Absolute Difference (MAD): {mad:.6f}")
    print(f"Root Mean Square Difference (RMSD): {rmsd:.6f}")
    print(f"Pearson Correlation: {pearson_corr:.6f}")
    print(f"Spearman Rank Correlation: {spearman_corr:.6f}")
    print(f"Paired t-test p-value: {ttest_result.pvalue:.6f}")

    # scatter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(reference, comparison, alpha=0.5, s=10)
    max_val = max(reference.max(), comparison.max())
    min_val = min(reference.min(), comparison.min())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 0].set_xlabel("Reference")
    axes[0, 0].set_ylabel("Comparison")
    axes[0, 0].set_title("Scatter Plot")

    # difference histogram
    axes[0, 1].hist(diff, bins=50, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Difference (Yours - Other)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Differences')

    # Cumulative distribution plot
    sorted_yours = np.sort(reference)
    sorted_other = np.sort(comparison)
    axes[1, 1].plot(sorted_yours, np.linspace(
        0, 1, len(sorted_yours)), label='Yours')
    axes[1, 1].plot(sorted_other, np.linspace(
        0, 1, len(sorted_other)), label='Other')
    axes[1, 1].set_xlabel('Expression Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Functions')
    axes[1, 1].legend()

    plt.show()


compare_with = sys.argv[1]
compare_csv(compare_with)
