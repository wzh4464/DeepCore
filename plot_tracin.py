import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress specific warnings related to NaN values in plots
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define directories to analyze
directories = [
    "results/flip_TracIn_1716_40",
    "results/flip_TracIn_1717_30",
    "results/flip_TracIn_1718_20", 
    "results/flip_TracIn_1719_10"
]

# Function to analyze a single directory
def analyze_directory(directory):
    print(f"\n==== Analyzing {directory} ====")
    
    # Read CSV files
    scores_path = os.path.join(directory, "flip_scores_0.csv")
    flipped_indices_path = os.path.join(directory, "flipped_indices.csv")
    
    scores = pd.read_csv(scores_path, header=None)
    flipped_indices = pd.read_csv(flipped_indices_path, header=None)
    
    # Assign column names
    scores.columns = ["score"]
    flipped_indices.columns = ["index"]
    
    # Print the original data
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
    print(f"Overlap percentage: {len(overlap) / len(flipped_set) * 100:.2f}%")
    print(f"Overlapping indices: {sorted(list(overlap))}")
    
    # Identify indices in one set but not the other
    only_in_top_n = top_n_set - flipped_set
    only_in_flipped = flipped_set - top_n_set
    print(f"\nIndices in top N but not in flipped_set: {len(only_in_top_n)}")
    print(f"Indices in flipped_set but not in top N: {len(only_in_flipped)}")
    
    # Calculate scores for histogram analysis (using the corrected indices)
    all_scores = scores["score"].values
    
    # Get scores for flipped indices (using corrected indices)
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
        "flip_percentage": int(directory.split("_")[-1]),
        "total_samples": len(scores),
        "flipped_samples": len(flipped_indices),
        "overlap_count": len(overlap),
        "overlap_percentage": len(overlap) / len(flipped_set) * 100,
        "flipped_set": flipped_set,  # Pass corrected flipped_set to the outer scope
    }, all_scores, flipped_scores, non_flipped_scores

# Process all directories
results = []
for directory in directories:
    if os.path.exists(directory):
        try:
            stats, all_scores, flipped_scores, non_flipped_scores = analyze_directory(directory)
            results.append(stats)
            
            # Create visualizations for each directory
            plt.figure(figsize=(12, 6))
            
            # Plot histograms of scores
            plt.subplot(1, 2, 1)
            if len(flipped_scores) > 0:
                plt.hist(flipped_scores, bins=min(30, len(flipped_scores)), alpha=0.7, label='Flipped')
            if len(non_flipped_scores) > 0:
                plt.hist(non_flipped_scores, bins=min(30, len(non_flipped_scores)), alpha=0.7, label='Non-flipped')
            plt.xlabel('TracIn Score')
            plt.ylabel('Count')
            plt.title(f'Distribution of TracIn Scores ({stats["flip_percentage"]}% Flipped)')
            plt.legend()
            
            # Plot sorted scores with flipped samples highlighted
            plt.subplot(1, 2, 2)
            
            # Get the indices from the sorted scores
            sorted_indices = np.argsort(all_scores)
            sorted_scores = all_scores[sorted_indices]
            
            # Create a mapping from array position to original index
            scores_df = pd.DataFrame({'score': all_scores})
            sorted_scores_df = scores_df.sort_values('score')
            
            # Check if each index in the sorted DataFrame is in the flipped set
            flipped_set = stats["flipped_set"]  # Get corrected flipped_set from the stats dictionary
            is_flipped = [idx in flipped_set for idx in sorted_scores_df.index]
            
            plt.scatter(range(len(sorted_scores)), sorted_scores_df['score'].values, 
                       c=['red' if f else 'blue' for f in is_flipped], 
                       alpha=0.5, s=5)
            plt.xlabel('Sorted Sample Index')
            plt.ylabel('TracIn Score')
            plt.title('Sorted TracIn Scores (Red = Flipped)')
            
            plt.tight_layout()
            plt.savefig(f'tracin_distribution_{stats["flip_percentage"]}pct.png')
            plt.close()
            
        except Exception as e:
            print(f"Error processing {directory}: {e}")
    else:
        print(f"Directory not found: {directory}")

# Create a summary table if we have results
if results:
    summary_df = pd.DataFrame(results)
    display_cols = [
        "directory", "flip_percentage", "total_samples", "flipped_samples", 
        "overlap_count", "overlap_percentage"
    ]
    summary_table = summary_df[display_cols].sort_values("flip_percentage")
    
    print("\n==== Summary of TracIn Analysis ====")
    print(summary_table)
    
    # Plot overlap percentage vs flip percentage
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["flip_percentage"], summary_df["overlap_percentage"], 'o-', linewidth=2, markersize=10)
    plt.xlabel('Flip Percentage (%)')
    plt.ylabel('Overlap Percentage (%)')
    plt.title('TracIn Detection Rate vs Flip Percentage')
    plt.grid(True)
    plt.xticks(summary_df["flip_percentage"])
    plt.savefig('tracin_detection_rate.png')
    plt.close()
else:
    print("No valid results found.")