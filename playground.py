# %%
import pandas as pd

# Read CSV files without header
scores = pd.read_csv("results/flip_TracIn_1697_30/flip_scores_0.csv")
flipped_indices = pd.read_csv("results/flip_TracIn_1697_30/flipped_indices.csv", header=None)

# results/flip_TracIn_1696_40
# results/flip_TracIn_1697_30
# results/flip_TracIn_1698_20
# results/flip_TracIn_1699_10

# Assign column name
scores.columns = ["score"]
flipped_indices.columns = ["index"]

# Remove scores items where score is 0.0
scores = scores[scores["score"] != 0.0]
scores
flipped_indices

# %%

# Sort scores in ascending order
sorted_scores = scores.sort_values("score", ascending=True)

# Get the top N indices where N is the length of flipped_set
top_n = sorted_scores.head(len(flipped_indices)).index

# Convert to sets for intersection
top_n_set = set(top_n)
flipped_set = set(flipped_indices["index"])

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
# %%
