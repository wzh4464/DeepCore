###
# File: ./deepcore/datasets/flipped_dataset.py
# Created Date: Friday, November 22nd 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 23rd November 2024 11:06:43 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import torch
import numpy as np


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]
        return data, target, real_idx

    def __len__(self):
        return len(self.indices)


class FlippedDataset(IndexedDataset):
    def __init__(self, dst_train, indices, num_scores, num_flip, dataset, seed, logger):
        super().__init__(dst_train, indices)
        self.num_scores = num_scores
        self.num_flip = num_flip
        self.data = dataset
        self.seed = seed
        self.logger = logger
        self.classes = dst_train.classes

        if self.data not in ["mnist", "MNIST"]:
            raise ValueError(f"Dataset {self.data} is not supported.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 根据排列后的索引获取标签为1的样本
        self.one_indices = [
            i for i, idx in enumerate(self.indices) if self.dataset.targets[idx] == 1
        ]

        self.flipped_indices = np.random.choice(
            self.one_indices, size=self.num_flip, replace=False
        )
        self.flipped_indices_set = set(self.flipped_indices.tolist())

        # 创建新的targets列表
        self.targets = self.dataset.targets.clone()
        for idx in self.flipped_indices:
            self.targets[self.indices[idx]] = 7

        self.logger.info(f"Flipped {self.num_flip} labels from 1 to 7.")
        self.logger.info(f"Flipped indices: {self.flipped_indices}")

    def get_flipped_indices(self):
        """
        Retrieve the indices of the flipped dataset.

        Returns:
            list: A list of indices representing the flipped dataset.
        """
        return self.flipped_indices
