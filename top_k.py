###
# File: ./top_k.py
# Created Date: Saturday, November 23rd 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 23rd November 2024 5:22:58 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np

# 读取CSV文件
scores = np.loadtxt("results/flip_oti_272_30/flip_scores_0.csv")

# 获取最小的40个值的索引
# argsort 会返回按值从小到大排序的索引
smallest_40_indices = np.argsort(scores)[:40]

# 获取这些索引对应的值
smallest_40_values = scores[smallest_40_indices]

# 打印结果
print("最小40个值的行号（从0开始）及其对应的值：")
for idx, value in zip(smallest_40_indices, smallest_40_values):
    print(f"行号: {idx}, 值: {value}")



from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
first_image = mnist.test.images[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
