# %%
# 导入依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# 读取 CSV 文件
scores = pd.read_csv("results/test_oti/average_score_2025-05-09_19-23-23.csv")
selection_from = pd.read_csv("results/test_oti/flipped_selection_from_2025-05-09_19-47-56_0.csv")
flipped_indices = pd.read_csv("results/test_oti/flipped_indices.csv", header=None)

# %%
# 预处理数据：重命名列名
flipped_indices.columns = ["index"]
scores.rename(columns={"0": "score"}, inplace=True)
selection_from.rename(columns={"0": "index"}, inplace=True)

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
