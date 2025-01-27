###
# File: ./liveval/datasets/ corrupted_dataset.py
# Created Date: Tuesday, November 26th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 27th January 2025 4:50:58 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# 文件路径：./liveval/datasets/corrupted_dataset.py

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from .flipped_dataset import IndexedDataset, FlippedDataset


class CorruptedDataset(FlippedDataset):
    def __init__(
        self, dataset, indices, num_scores, num_corrupt, dataset_name, seed, logger
    ):
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets
        self.num_scores = num_scores
        self.num_corrupt = num_corrupt
        self.data = dataset_name.lower()
        self.seed = seed
        self.logger = logger

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 定义高斯模糊变换
        self.corruption = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5.0,5.0))

        # 从数据集中随机选择需要进行模糊处理的样本索引
        self.corrupted_indices_permuted = np.random.choice(
            self.indices, size=self.num_corrupt, replace=False
        )
        self.corrupted_indices_unpermuted = [
            self.indices[idx] for idx in self.corrupted_indices_permuted
        ]

        # 初始化 scores_indices，包括被模糊的样本索引
        self.scores_indices = self.corrupted_indices_unpermuted.copy()

        # 计算还需要多少额外的样本来达到 num_scores 的数量
        remaining_indices = list(
            set(self.indices) - set(self.corrupted_indices_unpermuted)
        )
        additional_needed = self.num_scores - len(self.corrupted_indices_unpermuted)

        if additional_needed > 0 and len(remaining_indices) > 0:
            # 随机选择额外的样本
            additional_indices = np.random.choice(
                remaining_indices,
                size=min(additional_needed, len(remaining_indices)),
                replace=False,
            )
            self.scores_indices.extend(additional_indices)
            
        if self.data.lower() == "mnist":
            self.classes = dataset.classes

        self.logger.info(
            f"Corrupted {len(self.corrupted_indices_unpermuted)} inputs with Gaussian blur."
        )
        self.logger.info(f"Corrupted indices: {self.corrupted_indices_unpermuted}")

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]

        # 如果样本在被模糊的索引中，则对其进行高斯模糊处理
        if real_idx in self.corrupted_indices_unpermuted:
            data = self.corruption(data)

        return data, target, torch.tensor(real_idx, dtype=torch.long)

    def get_corrupted_indices(self):
        """获取被模糊处理的样本的真实索引"""
        return self.corrupted_indices_unpermuted

    def get_corrupted_selection_from(self):
        """获取用于计算分数的样本的真实索引"""
        return self.scores_indices

    def get_flipped_selection_from(self):
        """获取用于计算分数的样本的真实索引"""
        return self.get_corrupted_selection_from()

    def get_flipped_indices(self):
        """获取被模糊处理的样本的真实索引"""
        return self.get_corrupted_indices()
