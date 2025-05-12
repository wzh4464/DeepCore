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
from liveval.datasets.flipped_dataset import IndexedDataset

class BoundaryMNISTDataset(IndexedDataset):
    def __init__(self, dst_train, indices, num_scores, num_boundary, seed, logger, 
                 thickness_range=(1, 3), intensity_range=(0.8, 1.2), 
                 slant_range=(-0.3, 0.3)):
        """
        初始化边界点MNIST数据集
        
        参数:
            dst_train: 原始MNIST数据集
            indices: 使用的索引
            num_scores: 计算分数的总样本数
            num_boundary: 生成边界点的数量
            seed: 随机种子
            logger: 日志对象
            thickness_range: 笔画粗细变化范围
            intensity_range: 强度变化范围
            slant_range: 倾斜角度变化范围
        """
        super().__init__(dst_train, indices)
        self.num_scores = num_scores
        self.num_boundary = num_boundary
        self.seed = seed
        self.logger = logger
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self._select_boundary_candidates()
        self._generate_boundary_points(thickness_range, intensity_range, slant_range)
        self.logger.info(f"生成了 {len(self.boundary_indices)} 个边界点")
    
    def _select_boundary_candidates(self):
        self.candidate_indices = np.random.choice(
            self.indices, size=self.num_boundary, replace=False
        )
        self.boundary_indices = []
        self.boundary_images = {}
        self.boundary_transforms = {}
    
    def _generate_boundary_points(self, thickness_range, intensity_range, slant_range):
        try:
            from morphomnist import morpho, perturb
        except ImportError:
            self.logger.warning("未安装morphomnist，边界点生成将跳过。请先安装morphomnist。"); return
        for idx in self.candidate_indices:
            try:
                image, label = self._get_original_image(idx)
                transform_type = np.random.choice([
                    'thickness', 'intensity', 'slant', 'combined'
                ])
                if transform_type == 'thickness':
                    thickness = np.random.uniform(*thickness_range)
                    transformed = morpho.ImageMorphology(image.squeeze().numpy())
                    transformed = transformed.dilate(thickness)
                    new_image = torch.tensor(transformed.astype(float)).unsqueeze(0)
                    transform_params = {'thickness': thickness}
                elif transform_type == 'intensity':
                    intensity = np.random.uniform(*intensity_range)
                    new_image = image * intensity
                    transform_params = {'intensity': intensity}
                elif transform_type == 'slant':
                    slant = np.random.uniform(*slant_range)
                    transformed = perturb.Perturbation(image.squeeze().numpy())
                    transformed = transformed.slant(slant)
                    new_image = torch.tensor(transformed.astype(float)).unsqueeze(0)
                    transform_params = {'slant': slant}
                else:
                    thickness = np.random.uniform(*thickness_range)
                    intensity = np.random.uniform(*intensity_range)
                    slant = np.random.uniform(*slant_range)
                    transformed = morpho.ImageMorphology(image.squeeze().numpy())
                    transformed = transformed.dilate(thickness)
                    transformed = perturb.Perturbation(transformed)
                    transformed = transformed.slant(slant)
                    new_image = torch.tensor(transformed.astype(float)).unsqueeze(0) * intensity
                    transform_params = {
                        'thickness': thickness,
                        'intensity': intensity,
                        'slant': slant
                    }
                self.boundary_images[idx] = new_image
                self.boundary_transforms[idx] = {
                    'type': transform_type,
                    'params': transform_params
                }
                self.boundary_indices.append(idx)
            except Exception as e:
                self.logger.warning(f"生成边界点 {idx} 时出错: {str(e)}")
        self.scores_indices = self.boundary_indices.copy()
        remaining = list(set(self.indices) - set(self.boundary_indices))
        additional_needed = self.num_scores - len(self.boundary_indices)
        if additional_needed > 0 and remaining:
            additional = np.random.choice(
                remaining,
                size=min(additional_needed, len(remaining)),
                replace=False
            )
            self.scores_indices.extend(additional)
    def _get_original_image(self, idx):
        data, target = self.dataset[idx][:2]
        return data, target
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        result = self.dataset[real_idx]
        if len(result) == 3:
            data, target, _ = result
        else:
            data, target = result
        if real_idx in self.boundary_indices:
            data = self.boundary_images[real_idx]
        return data, target, torch.tensor(real_idx, dtype=torch.long)
    def get_boundary_indices(self):
        return self.boundary_indices
    def get_boundary_selection_from(self):
        return self.scores_indices 