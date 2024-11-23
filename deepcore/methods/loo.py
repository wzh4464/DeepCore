###
# File: ./deepcore/methods/loo.py
# Created Date: Thursday, November 21st 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 23rd November 2024 1:21:57 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# FILE: loo.py

import concurrent
from typing import override
from .influence_method import InfluenceMethod
from .selection_methods import SELECTION_METHODS
import torch
import torch.multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # 新增导入


class LOO(InfluenceMethod):
    """
    Leave-One-Out 方法，通过依次移除每个数据点，评估其对测试集损失的影响。
    """

    def select(self, **kwargs):
        influence_scores = self.compute_influence_scores()
        selected_indices = np.argsort(influence_scores)[::-1][: self.coreset_size]

        self.logger.info(f"Selected indices: {selected_indices}")

        self.logger.info(f"Selected scores: {influence_scores}")
        # save loss values to csv file
        np.savetxt(
            f"{self.args.save_path}/loo_losses.csv",
            influence_scores,
            delimiter=",",
            fmt="%0.6f",
        )

        return {"indices": selected_indices, "scores": influence_scores}

    def compute_influence_scores(self):
        num_scores = self.num_scores
        losses = np.zeros(num_scores)

        # 使用多个GPU进行并行计算
        num_gpus = self.num_gpus
        gpu_indices = np.array_split(np.arange(num_scores), num_gpus)

        # 使用 ProcessPoolExecutor 并行处理
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, indices in enumerate(gpu_indices):
                device = i  # 每个进程绑定到不同的 GPU
                futures.append(executor.submit(self.process_indices, indices, device))

            results = [
                future.result()
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="处理GPU任务",
                )
            ]
        # 汇总所有 GPU 的结果
        for i, indices in enumerate(gpu_indices):
            losses[indices] = results[i]

        # 计算完整模型的损失
        full_model = self.train_model(
            self.dst_train, device_id=0
        )  # 使用第一个 GPU 或其他策略
        full_loss = self.evaluate_model(full_model, device_id=0)
        return full_loss - losses

    def process_indices(self, indices, device_id):
        """
        处理分配给特定 GPU 的样本索引，计算每个移除样本后的测试集损失。
        """
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        local_losses = np.zeros(len(indices))
        self.logger.info(
            f"Process {torch.multiprocessing.current_process().name} using GPU {device}"
        )
        for i, idx in enumerate(indices):
            subset_indices = list(range(len(self.dst_train)))
            subset_indices.pop(idx)
            subset_dataset = torch.utils.data.Subset(self.dst_train, subset_indices)
            model = self.train_model(subset_dataset, device_id)
            loss = self.evaluate_model(model, device_id)
            local_losses[i] = loss
            self.logger.info(
                f"GPU {device_id}: Processed sample {idx+1}/{len(self.dst_train)}, Loss: {loss} for {i}th sample"
            )
        return local_losses
 
    @override
    def get_scores(self):
        return self.compute_influence_scores()


SELECTION_METHODS["loo"] = LOO
