###
# File: ./order.py
# Created Date: Saturday, November 23rd 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 23rd November 2024 5:28:35 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
from scipy import stats

# 读取两个文件
scores_30 = np.loadtxt("results/flip_oti_272_30/flip_scores_0.csv")
scores_40 = np.loadtxt("results/flip_oti_273_40/flip_scores_0.csv")

# 获取每个数组的排序索引 从大到小
order_30 = np.argsort(scores_30)
order_40 = np.argsort(scores_40)

# 计算 Spearman 相关系数
correlation, p_value = stats.spearmanr(scores_30, scores_40)


# 计算前N个最小值的重叠
def overlap_percentage(arr1, arr2, N):
    set1 = set(arr1[:N])
    set2 = set(arr2[:N])
    overlap = len(set1.intersection(set2))
    return (overlap / N) * 100


# 检查不同N值的重叠度
N_values = [10, 20, 30, 40, 50, 100]
for N in N_values:
    overlap = overlap_percentage(order_30, order_40, N)
    print(f"前{N}个索引的重叠度: {overlap:.2f}%")

print(f"\nSpearman 相关系数: {correlation:.4f}")
print(f"P-value: {p_value:.4e}")

# 打印两个文件中共同的前N个索引
N = 40  # 可以根据需要调整
common_indices = set(order_30[:N]).intersection(set(order_40[:N]))
print(f"\n在前{N}个索引中的共同索引:")
print(sorted(list(common_indices)))

# 打印差异
only_in_30 = set(order_30[:N]) - set(order_40[:N])
only_in_40 = set(order_40[:N]) - set(order_30[:N])
print(f"\n只在30中的前{N}个索引: {sorted(list(only_in_30))}")
print(f"只在40中的前{N}个索引: {sorted(list(only_in_40))}")
