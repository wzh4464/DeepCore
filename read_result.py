###
# File: /read_result.py
# Created Date: Monday, November 11th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 12th November 2024 9:32:52 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_selection_result(file_path):
    """
    Analyze selection result from .pt file

    Args:
        file_path (str): Path to selection_result.pt file
    """
    # Load the result file
    result = torch.load(file_path)

    # Extract data
    indices = result["indices"]
    scores = result["scores"]
    time_valuations = result["time_valuations"]

    print("\nBasic Statistics:")
    print(f"Number of selected samples: {len(indices)}")
    print("Score statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    print(f"  Min: {np.min(scores):.4f}")
    print(f"  Max: {np.max(scores):.4f}")

    # Plot score distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=50, edgecolor="black")
    _setup_chart_labels("Score Distribution", "Score", "Count")
    # Plot time series valuations
    plt.subplot(1, 2, 2)
    time_series_mean = np.array([v.mean() for v in time_valuations])
    time_series_std = np.array([v.std() for v in time_valuations])

    plt.plot(time_series_mean, label="Mean valuation")
    plt.fill_between(
        range(len(time_series_mean)),
        time_series_mean - time_series_std,
        time_series_mean + time_series_std,
        alpha=0.3,
    )
    _setup_chart_labels("Time Series Valuations", "Time Step", "Mean Valuation")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return result


def _setup_chart_labels(chart_title, x_axis_label, y_axis_label):
    plt.title(chart_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)


# Example usage
if __name__ == "__main__":
    import sys

    # read from command line
    try:
        file_path = sys.argv[1]
    except Exception:
        file_path = "/backup/adoti_results_test_small/selection_result.pt"

    result = analyze_selection_result(file_path)

    # You can now access individual components
    indices = result["indices"]
    scores = result["scores"]
    time_valuations = result["time_valuations"]

    # Additional analysis as needed
    print("\nTop 10 indices and their scores:")
    for idx, score in zip(indices[:10], scores[indices[:10]]):
        print(f"Index: {idx}, Score: {score:.4f}")
