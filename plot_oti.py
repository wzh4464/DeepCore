import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob
from matplotlib.ticker import MaxNLocator
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
    overlap_percentage = len(overlap) / len(flipped_set) * 100 if len(flipped_set) > 0 else 0
    print(f"Overlap percentage: {overlap_percentage:.2f}%")
    
    # Calculate scores for histogram analysis
    oti_scores = scores["score"].values
    
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
    
    return {
        "directory": directory,
        "seed": seed,
        "flip_percentage": flip_percentage,
        "total_samples": len(scores),
        "flipped_samples": len(flipped_indices),
        "overlap_count": len(overlap),
        "overlap_percentage": overlap_percentage,
    }, oti_scores, flipped_scores, non_flipped_scores

# Function to get oti experiment directories
def get_oti_result_dirs():
    # Pattern to match oti Oti directories with flip percentages
    pattern = "results/flip_Oti_*"
    return glob.glob(pattern)

# Main function
def analyze_oti_results():
    print("Analyzing oti Oti experiment results...")
    
    # Get oti directories
    oti_directories = get_oti_result_dirs()
    print(f"Found {len(oti_directories)} potential result directories")
    
    oti_results = []
    
    # Process oti directories
    for directory in oti_directories:
        stats, oti_scores, flipped_scores, non_flipped_scores = analyze_directory(directory)
        
        if stats is not None:
            oti_results.append(stats)
            
            # Create visualizations for each directory if needed
            if len(flipped_scores) > 0 and len(non_flipped_scores) > 0:
                plt.figure(figsize=(12, 6))
                
                # Plot histograms of scores
                plt.subplot(1, 2, 1)
                plt.hist(flipped_scores, bins=min(30, len(flipped_scores)), alpha=0.7, label='Flipped')
                plt.hist(non_flipped_scores, bins=min(30, len(non_flipped_scores)), alpha=0.7, label='Non-flipped')
                plt.xlabel('Oti Score')
                plt.ylabel('Count')
                plt.title(f'Distribution of Oti Scores ({stats["flip_percentage"]}% Flipped, Seed {stats["seed"]})')
                plt.legend()
                
                # Ensure the output directory exists
                os.makedirs('plots', exist_ok=True)
                
                plt.tight_layout()
                plt.savefig(f'plots/oti_distribution_{stats["flip_percentage"]}pct_seed{stats["seed"]}.png')
                plt.close()
    
    # Create a summary table if we have results
    if oti_results:
        summary_df = pd.DataFrame(oti_results)
        display_cols = [
            "directory", "seed", "flip_percentage", "total_samples", "flipped_samples", 
            "overlap_count", "overlap_percentage"
        ]
        summary_table = summary_df[display_cols].sort_values(["flip_percentage", "seed"])
        
        print("\n==== Summary of OTI Analysis Across Seeds ====")
        print(summary_table)
        
        # Save the summary table
        summary_table.to_csv('oti_summary_oti.csv', index=False)
        
        # Create summary statistics grouped by flip percentage
        grouped_stats = summary_df.groupby("flip_percentage").agg({
            "overlap_percentage": ["mean", "std", "min", "max", "count"]
        })
        
        # Flatten the column names
        grouped_stats.columns = ["_".join(col).strip() for col in grouped_stats.columns.values]
        grouped_stats = grouped_stats.reset_index()
        
        print("\n==== Summary Statistics by Flip Percentage ====")
        print(grouped_stats)
        
        # Save the grouped statistics
        grouped_stats.to_csv('oti_grouped_stats_oti.csv', index=False)
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # ===============================================
        # Plot 1: Line plot with error bars
        # ===============================================
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            grouped_stats["flip_percentage"], 
            grouped_stats["overlap_percentage_mean"], 
            yerr=grouped_stats["overlap_percentage_std"],
            fmt='o-', linewidth=2, markersize=10, capsize=5
        )
        plt.xlabel('Flip Percentage (%)', fontsize=12)
        plt.ylabel('Detection Rate (%)', fontsize=12)
        plt.title('Average Oti Detection Rate vs Flip Percentage (Across Seeds)', fontsize=14)
        plt.grid(True)
        plt.xticks(grouped_stats["flip_percentage"])
        
        # Add value labels above each point
        for i, row in grouped_stats.iterrows():
            plt.text(row["flip_percentage"], 
                    row["overlap_percentage_mean"] + row["overlap_percentage_std"] + 2, 
                    f'{row["overlap_percentage_mean"]:.1f}%', 
                    ha='center', fontweight='bold')
        
        # Add min/max range
        for i, row in grouped_stats.iterrows():
            plt.plot([row["flip_percentage"], row["flip_percentage"]], 
                     [row["overlap_percentage_min"], row["overlap_percentage_max"]], 'r-', alpha=0.3)
            
            # Add min/max labels
            plt.text(row["flip_percentage"] + 0.5, row["overlap_percentage_min"] - 1, 
                     f'Min: {row["overlap_percentage_min"]:.1f}%', fontsize=8)
            plt.text(row["flip_percentage"] + 0.5, row["overlap_percentage_max"] + 1, 
                     f'Max: {row["overlap_percentage_max"]:.1f}%', fontsize=8)
        
        plt.savefig('plots/oti_detection_rate_oti.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ===============================================
        # Plot 2: Box plots for each flip percentage
        # ===============================================
        plt.figure(figsize=(12, 8))
        
        # Create box plot data
        data = []
        labels = []
        for pct in sorted(summary_df["flip_percentage"].unique()):
            values = summary_df[summary_df["flip_percentage"] == pct]["overlap_percentage"].values
            data.append(values)
            labels.append(f'{pct}%')
        
        # Create box plot with updated parameter name
        plt.boxplot(data, tick_labels=labels, patch_artist=True)
        
        # Add individual data points
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.05, size=len(d))
            plt.scatter(x, d, alpha=0.5, c='black', s=30)
        
        plt.xlabel('Flip Percentage (%)', fontsize=12)
        plt.ylabel('Detection Rate (%)', fontsize=12)
        plt.title('Distribution of Oti Detection Rates Across Seeds', fontsize=14)
        plt.grid(True, axis='y')
        
        # Add mean values on top of each box
        for i, d in enumerate(data):
            plt.text(i+1, np.max(d) + 2, f'Mean: {np.mean(d):.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.savefig('plots/oti_detection_boxplot_oti.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ===============================================
        # Plot 3: Swarm plot (alternative visualization)
        # ===============================================
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        ax = sns.swarmplot(x="flip_percentage", y="overlap_percentage", data=summary_df, size=10)
        ax = sns.boxplot(x="flip_percentage", y="overlap_percentage", data=summary_df, 
                         showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", 
                                                  "markeredgecolor":"black", "markersize":"10"})
        
        plt.xlabel('Flip Percentage (%)', fontsize=12)
        plt.ylabel('Detection Rate (%)', fontsize=12)
        plt.title('Distribution of Oti Detection Rates by Flip Percentage', fontsize=14)
        
        # Add value annotations for mean
        for i, pct in enumerate(sorted(summary_df["flip_percentage"].unique())):
            subset = summary_df[summary_df["flip_percentage"] == pct]
            mean_val = subset["overlap_percentage"].mean()
            plt.text(i, mean_val + 2, f'Mean: {mean_val:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.savefig('plots/oti_detection_swarm_oti.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ===============================================
        # Plot 4: Heatmap of detection rates by seed and flip percentage
        # ===============================================
        plt.figure(figsize=(12, 8))
        
        # Pivot the data to create a matrix of seeds vs flip percentages
        pivot_df = summary_df.pivot(index="seed", columns="flip_percentage", values="overlap_percentage")
        
        # Create the heatmap
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5)
        
        plt.title('Detection Rate (%) by Seed and Flip Percentage', fontsize=14)
        plt.xlabel('Flip Percentage (%)', fontsize=12)
        plt.ylabel('Seed', fontsize=12)
        
        plt.savefig('plots/oti_detection_heatmap_oti.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ===============================================
        # Plot 5: Combined plot - side by side box plots for each flip percentage
        # ===============================================
        flip_percentages = sorted(summary_df["flip_percentage"].unique())
        num_flips = len(flip_percentages)
        
        if num_flips > 1:  # Only create this plot if we have more than one flip percentage
            fig, axes = plt.subplots(1, num_flips, figsize=(5*num_flips, 8), sharey=True)
            
            for i, pct in enumerate(flip_percentages):
                subset = summary_df[summary_df["flip_percentage"] == pct]
                
                # Box plot - handle case with single or multiple axes
                if num_flips > 1:
                    ax = axes[i]
                else:
                    ax = axes
                    
                bp = ax.boxplot(subset["overlap_percentage"], patch_artist=True)
                
                # Individual points
                y = subset["overlap_percentage"].values
                x = np.random.normal(1, 0.05, size=len(y))
                ax.scatter(x, y, alpha=0.5, c='black', s=40)
                
                # Styling
                ax.set_title(f'{pct}% Flipped', fontsize=12)
                ax.set_xlabel('Seeds', fontsize=10)
                ax.grid(True, axis='y')
                
                # Add mean value
                mean_val = subset["overlap_percentage"].mean()
                max_val = np.max(y) if len(y) > 0 else 0
                ax.text(1, max_val + 2, f'Mean: {mean_val:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
                
                # Only add y-label for the first plot
                if i == 0:
                    ax.set_ylabel('Detection Rate (%)', fontsize=12)
            
            plt.suptitle('Oti Detection Rates by Flip Percentage', fontsize=16)
            plt.tight_layout()
            plt.savefig('plots/oti_detection_combined_oti.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    else:
        print("No valid results found.")

# Run the analysis
if __name__ == "__main__":
    analyze_oti_results()