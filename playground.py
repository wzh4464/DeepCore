# %%
# 导入依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# 读取 CSV 文件
scores = pd.read_csv("results/boundary_detection/average_boundary_score_2025-05-12_16-25-17.csv")
selection_from = pd.read_csv("results/boundary_detection/boundary_indices_2025-05-12_16-25-17_0.csv")
flipped_indices = pd.read_csv("./results/boundary_detection/boundary_indices.csv", header=None)

# %%
# 预处理数据：重命名列名
flipped_indices.columns = ["index"]
scores.rename(columns={"0": "score"}, inplace=True)
selection_from.rename(columns={"boundary_indices": "index"}, inplace=True)

# %%
# 排序并获取 top N
sorted_scores = scores.sort_values("score", ascending=True)
top_n = sorted_scores.head(len(flipped_indices)).index

# %%
# 打印各类排序后的数据，带表头
print("Top N indices (按分数升序排序):")
print(top_n.sort_values())
print("\nFlipped indices (按index升序排序):")
print(flipped_indices.sort_values("index"))
print("\nSelection from (按index升序排序):")
print(selection_from.sort_values("index"))
print("\nScores (按分数升序排序):")
print(scores.sort_values("score", ascending=True))

# %%
# 计算交集与差集
# 转为集合
top_n_set = set(top_n)
flipped_set = set(flipped_indices["index"])

# 找到重叠部分
overlap = top_n_set.intersection(flipped_set)

# 展示结果
print(f"Number of overlapping indices: {len(overlap)}")
print(f"Overlap percentage: {len(overlap) / len(flipped_set) * 100:.2f}%")
print(f"Overlapping indices: {sorted(list(overlap))}")

# 只在 top_n 或 flipped_set 中的索引
only_in_top_n = top_n_set - flipped_set
only_in_flipped = flipped_set - top_n_set

print(f"\nIndices in top N but not in flipped_set: {len(only_in_top_n)}")
print(f"Indices in flipped_set but not in top N: {len(only_in_flipped)}")

# %%
# flipped_indices 在 selection_from 中的数量
print(f"Number of flipped indices in selection from: {len(selection_from[selection_from['index'].isin(flipped_indices['index'])])}")

# %%
# 比较所有有分数的索引和 selection_from 的索引是否有重叠
# 去掉 nan 分数
scores_indices = scores[~scores["score"].isna()].index
scores_indices_set = set(scores_indices)

print(f"Number of scores indices: {len(scores_indices_set)}")
# %%
selection_from_set = set(selection_from["index"])
overlap_indices = scores_indices_set.intersection(selection_from_set)

print(f"Number of overlap indices: {len(overlap_indices)}")

# %%
# 比较所有有分数的索引和 flipped_indices 的索引是否有重叠
scores_indices = scores[~scores["score"].isna()].index
scores_indices_set = set(scores_indices)

print(f"Number of scores indices: {len(scores_indices_set)}")

# %%
# 计算四个集合两两之间的重叠比例，并以矩阵表格输出
sets = [scores_indices_set, top_n_set, selection_from_set, flipped_set]
set_names = ["有分数的", "最低分数top_n", "selection_from", "flipped_indices"]

# 计算重叠比例矩阵（行是A，列是B，表示A中有多少比例在B中）
ratio_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if len(sets[i]) == 0:
            ratio_matrix[i, j] = np.nan
        else:
            ratio_matrix[i, j] = len(sets[i] & sets[j]) / len(sets[i])

# 输出表格
ratio_df = pd.DataFrame(ratio_matrix, index=set_names, columns=set_names)
print("\n四个集合两两之间的重叠比例矩阵（行是A，列是B，表示A中有多少比例在B中）：")
print(ratio_df.applymap(lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "nan"))

# %%
# 可视化 flipped_indices 在分数从低到高排序中的分布
# 去掉 nan 分数
scores_no_nan = scores[~scores["score"].isna()].copy()
# 按分数升序排序
scores_no_nan_sorted = scores_no_nan.sort_values("score", ascending=True)
# 获取排序后索引的列表
sorted_indices = list(scores_no_nan_sorted.index)
# flipped_indices 在排序后中的位置
flipped_pos = [sorted_indices.index(idx) for idx in flipped_indices["index"] if idx in sorted_indices]

plt.figure(figsize=(10, 4))
plt.hist(flipped_pos, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Location in sorted scores")
plt.ylabel("flipped_indices appearance")
plt.title("Distribution of flipped_indices in sorted scores")
plt.tight_layout()
plt.show()

# %%
# 在分数分布直方图中叠加 flipped_indices 的比例（蓝色）
plt.figure(figsize=(10, 4))

# 所有分数（去除 nan）
scores_all = scores_no_nan_sorted["score"].values
# flipped 分数（去除 nan 且在 scores 中）
flipped_scores = scores_no_nan_sorted.loc[scores_no_nan_sorted.index.isin(flipped_indices["index"]), "score"].values

bins = 30
counts, bin_edges, _ = plt.hist(scores_all, bins=bins, color="salmon", edgecolor="black", alpha=1, label="All samples")

# 统计每个 bin 内 flipped 的数量
flipped_counts, _ = np.histogram(flipped_scores, bins=bin_edges)

# 画出蓝色部分，表示 flipped 在每个区间的数量（高度与总样本一致，蓝色部分为 flipped 占比）
plt.bar(
    (bin_edges[:-1] + bin_edges[1:]) / 2,
    flipped_counts,
    width=(bin_edges[1] - bin_edges[0]),
    color="skyblue",
    alpha=0.8,
    label="Flipped in bin"
)

plt.xlabel("score")
plt.ylabel("Number of samples")
plt.title("score distribution histogram with flipped overlay")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# TracIn order stability

# print version of python and matplotlib
import sys
import matplotlib # For matplotlib.__version__
import matplotlib.pyplot as plt # For plotting functions like figure, xlabel, etc.
import pandas as pd # For DataFrame manipulation
import seaborn as sns # For seaborn plots

print(f"Python version: {sys.version}")
print(f"Matplotlib version: {matplotlib.__version__}")
# Optionally, print Seaborn version if desired for debugging or records
# print(f"Seaborn version: {sns.__version__}")

# print(f"Number of focus samples: {fouces_num}") # This was commented out in the original
tracin_scores_focus_first = pd.read_csv("results/tracin_order_stability/MNIST_LeNet_cuda_bak/seed42_e5_b256_s100/tracin_scores_focus_first.csv")
tracin_scores_focus_last = pd.read_csv("results/tracin_order_stability/MNIST_LeNet_cuda_bak/seed42_e5_b256_s100/tracin_scores_focus_last.csv")

plt.rcParams["font.family"] = "Times New Roman"

# --- First Plot: Original TracIn Scores ---

# Prepare data for Seaborn
# Filter for 'is_focus' (assuming it's a boolean column) and select 'tracin_score'
scores_ff_orig = tracin_scores_focus_first[tracin_scores_focus_first["is_focus"]]["tracin_score"]
scores_fl_orig = tracin_scores_focus_last[tracin_scores_focus_last["is_focus"]]["tracin_score"]

# Combine into a single DataFrame with a 'Condition' column for hue
df_plot_orig = pd.concat([
    pd.DataFrame({'tracin_score': scores_ff_orig, 'Condition': 'focus_first'}),
    pd.DataFrame({'tracin_score': scores_fl_orig, 'Condition': 'focus_last'})
])

# Create the plot
plt.figure(figsize=(10, 4))
sns.histplot(data=df_plot_orig, x="tracin_score", hue="Condition", 
             bins=20, 
             palette={"focus_first": "salmon", "focus_last": "skyblue"}, # Colors for each condition
             alpha=0.7, 
             edgecolor="black", 
             linewidth=1, 
             multiple="layer", # Overlay histograms, similar to how plt.hist([data1, data2]) behaves
             hue_order=["focus_first", "focus_last"]) # Ensure consistent order for colors and legend

plt.xlabel("TracIn Score", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Distribution of TracIn Scores: Focus First vs Focus Last", fontsize=14)
# plt.legend(fontsize=11) # Customize legend font size (Seaborn creates the legend via hue)
plt.tight_layout()
plt.show()

# --- Second Plot: Absolute TracIn Scores ---

# Prepare data for Seaborn (absolute values)
scores_ff_abs = tracin_scores_focus_first[tracin_scores_focus_first["is_focus"]]["tracin_score"].abs()
scores_fl_abs = tracin_scores_focus_last[tracin_scores_focus_last["is_focus"]]["tracin_score"].abs()

# Combine into a single DataFrame
df_plot_abs = pd.concat([
    pd.DataFrame({'tracin_score': scores_ff_abs, 'Condition': 'focus_first'}),
    pd.DataFrame({'tracin_score': scores_fl_abs, 'Condition': 'focus_last'})
])

# Create the plot
plt.figure(figsize=(10, 4))
sns.histplot(data=df_plot_abs, x="tracin_score", hue="Condition",
             bins=20,
             palette={"focus_first": "salmon", "focus_last": "skyblue"},
             alpha=0.7,
             edgecolor="black",
             linewidth=1,
             multiple="layer",
             hue_order=["focus_first", "focus_last"])

plt.xlabel("Absolute TracIn Score", fontsize=12) # X-axis label updated for absolute values
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Distribution of Absolute TracIn Scores: Focus First vs Focus Last", fontsize=14) # Title updated
# plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %%
