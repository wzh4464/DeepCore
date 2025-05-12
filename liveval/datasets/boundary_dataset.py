###
# File: ./liveval/datasets/boundary_dataset.py
# Created Date: Tuesday, May 13th 2025
# Author: Zihan
# -----
# Last Modified: Tuesday, 13th May 2025 10:15:23 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date          By      Comments
# ----------    ------  ---------------------------------------------------------
###

import torch
import numpy as np
from .flipped_dataset import IndexedDataset
import torchvision.transforms.functional as TF
import random

class BoundaryDataset(IndexedDataset):
    def __init__(
        self, 
        dst_train, 
        indices, 
        num_scores, 
        num_boundary, 
        dataset_name, 
        seed, 
        logger,
        transform_intensity=0.5  # 控制变形强度
    ):
        """
        初始化边界点数据集
        
        参数:
            dst_train: 原始数据集
            indices: 使用的索引
            num_scores: 计算分数的总样本数
            num_boundary: 生成边界点的数量
            dataset_name: 数据集名称
            seed: 随机种子
            logger: 日志对象
            transform_intensity: 变形强度
        """
        super().__init__(dst_train, indices)
        self.num_scores = num_scores
        self.num_boundary = num_boundary
        self.data = dataset_name.lower()
        self.seed = seed
        self.logger = logger
        self.transform_intensity = transform_intensity
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._setup_boundary_points()
    
    def _setup_boundary_points(self):
        self.boundary_indices_permuted = np.random.choice(
            len(self.indices), size=self.num_boundary, replace=False
        )
        self.boundary_indices_unpermuted = [
            self.indices[idx] for idx in self.boundary_indices_permuted
        ]
        self.transformed_images = {}
        self.transform_types = {}
        for idx in self.boundary_indices_unpermuted:
            data, target, _ = self.dataset[idx]
            transform_type = np.random.randint(1, 5)
            self.transform_types[idx] = transform_type
            transformed_image = self._apply_transform(data, transform_type)
            self.transformed_images[idx] = transformed_image
        self.scores_indices = self.boundary_indices_unpermuted.copy()
        remaining_indices = list(set(self.indices) - set(self.boundary_indices_unpermuted))
        additional_needed = self.num_scores - len(self.boundary_indices_unpermuted)
        if additional_needed > 0 and len(remaining_indices) > 0:
            additional_indices = np.random.choice(
                remaining_indices, 
                size=min(additional_needed, len(remaining_indices)), 
                replace=False
            )
            self.scores_indices.extend(additional_indices)
        if self.data.lower() == "mnist":
            self.classes = self.dataset.classes if hasattr(self.dataset, "classes") else None
        self.logger.info(
            f"生成了 {len(self.boundary_indices_unpermuted)} 个边界点。"
        )
    
    def _apply_transform(self, image, transform_type):
        intensity = self.transform_intensity
        if transform_type == 1:  # 高斯噪声
            noise = torch.randn_like(image) * intensity * 0.1
            transformed = torch.clamp(image + noise, 0, 1)
        elif transform_type == 2:  # 弹性变形（简化为旋转）
            angle = (random.random() * 2 - 1) * 15 * intensity
            transformed = TF.rotate(image, angle)
        elif transform_type == 3:  # 旋转
            angle = (random.random() * 2 - 1) * 20 * intensity
            transformed = TF.rotate(image, angle)
        elif transform_type == 4:  # 灰度缩放
            contrast_factor = 1 + (random.random() * 2 - 1) * intensity
            transformed = TF.adjust_contrast(image, contrast_factor)
        else:
            transformed = image
        return transformed
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        result = self.dataset[real_idx]
        if len(result) == 3:
            data, target, _ = result
        else:
            data, target = result
        if real_idx in self.boundary_indices_unpermuted:
            data = self.transformed_images[real_idx]
        return data, target, torch.tensor(real_idx, dtype=torch.long)
    
    def get_boundary_indices(self):
        return self.boundary_indices_unpermuted
    
    def get_boundary_selection_from(self):
        return self.scores_indices
    
    def get_flipped_indices(self):
        return self.get_boundary_indices()
    
    def get_flipped_selection_from(self):
        return self.get_boundary_selection_from() 