import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
import seaborn as sns

# Suppress specific warnings related to NaN values in plots
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Function to analyze a single directory
def analyze_directory(directory):
    print(f"\n==== Analyzing {directory} ====")

    # Extract seed and flip percentage from directory name
    parts = directory.split("_")
    seed_part = [p for p in parts if "seed" in p]
    seed = int(seed_part[0].replace("seed", "")) if seed_part else 0

    # Extract flip percentage from the directory name
    flip_percentage = None
    for i in range(len(parts)):
        try:
            flip_percentage = int(parts[i])
            # Check if this is actually the flip percentage (should be 10, 20, 30, or 40)
            if flip_percentage in [10, 20, 30, 40]:
                break
            else:
                flip_percentage = None
        except ValueError:
            continue

    if flip_percentage is None:
        print(f"Could not determine flip percentage for {directory}, skipping...")
        return None, None, None, None

    # Read CSV files
    scores_path = os.path.join(directory, "flip_scores_0.csv")
    flipped_indices_path = os.path.join(directory, "flipped_indices.csv")

    if not os.path.exists(scores_path) or not os.path.exists(flipped_indices_path):
        print(f"Required files not found in {directory}")
        return None, None, None, None

    scores = pd.read_csv(scores_path, header=None)
    flipped_indices = pd.read_csv(flipped_indices_path, header=None)

    # Assign column names
    scores.columns = ["score"]
    flipped_indices.columns = ["index"]

    # Print the original data
    print(f"Seed: {seed}, Flip percentage: {flip_percentage}")
    print(f"Original scores shape: {scores.shape}")
    print(f"Number of zeros in scores: {(scores['score'] == 0.0).sum()}")
    print(f"Flipped indices shape: {flipped_indices.shape}")

    # Remove scores items where score is 0.0
    scores = scores[scores["score"] != 0.0]
    print(f"After removing zeros, scores shape: {scores.shape}")

    # Sort scores in ascending order
    sorted_scores = scores.sort_values("score", ascending=True)

    # Get the top N indices where N is the length of flipped_set
    top_n = sorted_scores.head(len(flipped_indices)).index

    # Convert to sets for intersection
    top_n_set = set(top_n)

    # Create a corrected flipped set by adding 1 to each index to fix the off-by-one error
    flipped_set = set(index + 1 for index in flipped_indices["index"])

    # Find overlap
    overlap = top_n_set.intersection(flipped_set)

    # Show results
    print(f"Number of overlapping indices: {len(overlap)}")
    overlap_percentage = (
        len(overlap) / len(flipped_set) * 100 if len(flipped_set) > 0 else 0
    )
    print(f"Overlap percentage: {overlap_percentage:.2f}%")

    # Calculate scores for histogram analysis
    method_scores = scores["score"].values

    # Get scores for flipped indices
    flipped_scores = []
    for idx in flipped_set:
        if idx in scores.index:
            flipped_scores.append(scores.loc[idx, "score"])
    flipped_scores = np.array(flipped_scores)

    # Get scores for non-flipped indices
    non_flipped_indices = set(scores.index) - flipped_set
    non_flipped_scores = []
    for idx in non_flipped_indices:
        non_flipped_scores.append(scores.loc[idx, "score"])
    non_flipped_scores = np.array(non_flipped_scores)

    return (
        {
            "directory": directory,
            "seed": seed,
            "flip_percentage": flip_percentage,
            "total_samples": len(scores),
            "flipped_samples": len(flipped_indices),
            "overlap_count": len(overlap),
            "overlap_percentage": overlap_percentage,
        },
        method_scores,
        flipped_scores,
        non_flipped_scores,
    )


# Function to get experiment directories
def get_result_dirs(method):
    # Pattern to match directories with flip percentages
    pattern = f"results/flip_{method}_*"
    return glob.glob(pattern)


# Main function
def analyze_results(method="TracIn"):
    print(f"Analyzing {method} experiment results...")

    # Get directories
    result_directories = get_result_dirs(method)
    print(f"Found {len(result_directories)} potential result directories")

    results = []

    # Process directories
    for directory in result_directories:
        stats, method_scores, flipped_scores, non_flipped_scores = analyze_directory(
            directory
        )

        if stats is not None:
            results.append(stats)

    # Create a summary table if we have results
    if results:
        summary_df = pd.DataFrame(results)
        display_cols = [
            "directory",
            "seed",
            "flip_percentage",
            "total_samples",
            "flipped_samples",
            "overlap_count",
            "overlap_percentage",
        ]
        summary_table = summary_df[display_cols].sort_values(
            ["flip_percentage", "seed"]
        )

        print(f"\n==== Summary of {method} Analysis Across Seeds ====")
        print(summary_table)

        # Save the summary table
        summary_table.to_csv(f"plots/{method.lower()}_summary.csv", index=False)

        # Create summary statistics grouped by flip percentage
        grouped_stats = summary_df.groupby("flip_percentage").agg(
            {"overlap_percentage": ["mean", "std", "min", "max", "count"]}
        )

        # Flatten the column names
        grouped_stats.columns = [
            "_".join(col).strip() for col in grouped_stats.columns.values
        ]
        grouped_stats = grouped_stats.reset_index()

        print(f"\n==== Summary Statistics by Flip Percentage ====")
        print(grouped_stats)

        # Save the grouped statistics
        grouped_stats.to_csv(f"plots/{method.lower()}_grouped_stats.csv", index=False)

        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Plot: Line plot with shaded error regions
        plt.figure(figsize=(10, 6))

        # Set a nicer color palette
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
        sns.set_palette(colors)

        # Create the line plot
        plt.plot(
            grouped_stats["flip_percentage"],
            grouped_stats["overlap_percentage_mean"],
            "o-",
            linewidth=2,
            markersize=10,
            color=colors[0],
            label="Mean Detection Rate",
        )

        # Add shaded error region
        plt.fill_between(
            grouped_stats["flip_percentage"],
            grouped_stats["overlap_percentage_mean"]
            - grouped_stats["overlap_percentage_std"],
            grouped_stats["overlap_percentage_mean"]
            + grouped_stats["overlap_percentage_std"],
            alpha=0.3,
            color=colors[0],
            label="Standard Deviation",
        )

        # Add point labels
        for i, row in grouped_stats.iterrows():
            plt.text(
                row["flip_percentage"],
                row["overlap_percentage_mean"] + 2,
                f'{row["overlap_percentage_mean"]:.1f}%',
                ha="center",
                fontweight="bold",
                color=colors[0],
            )

        # Styling
        plt.xlabel("Flip Percentage (%)", fontsize=12)
        plt.ylabel("Detection Rate (%)", fontsize=12)
        plt.title(
            f"{method} Detection Rate vs Flip Percentage (Across Seeds)",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.xticks(grouped_stats["flip_percentage"])
        plt.legend(loc="lower right")

        # Set y-axis to start from 0
        plt.ylim(bottom=0)

        # Add a light background grid
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.savefig(
            f"plots/{method.lower()}_detection_rate_smooth.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        print("No valid results found.")


# Run the analysis
if __name__ == "__main__":
    # 可以选择 'TracIn', 'OTI', 'Grand', 'LOO'
    os.makedirs("plots", exist_ok=True)
    analyze_results(method="TracIn")
    analyze_results(method="OTI")
