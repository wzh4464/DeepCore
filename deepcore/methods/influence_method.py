###
# File: ./deepcore/methods/influence_method.py
# Created Date: Thursday, November 21st 2024
# Author: Zihan
# -----
# Last Modified: Friday, 22nd November 2024 10:41:53 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# FILE: influence_method.py

import logging
import torch
import numpy as np
import copy

from .coresetmethod import CoresetMethod
from .. import nets


class InfluenceMethod(CoresetMethod):
    """
    基类，用于通过移除数据点来评估其影响。
    """

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.model_class = args.model  # 模型类
        self.device = args.device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dst_test = kwargs.get("dst_test")
        self.logger = logging.getLogger(__name__)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # handler.setFormatter(formatter)
        # self.logger.addHandler(handler)
        self.num_scores = args.num_scores if hasattr(args, "num_scores") else 100
        self.num_gpus = args.num_gpus if hasattr(args, "num_gpus") else 1
        # 固定随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # if num_gpus > 1: then use spawn method to create new processes
        if self.num_gpus > 1:
            torch.multiprocessing.set_start_method("spawn")

    def train_model(self, train_dataset, device_id=0):
        """
        使用给定的训练数据集训练模型。
        """
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

        # 设置随机种子
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)

        model = nets.__dict__[self.model_class](
            self.args.channel,
            (self.num_classes),
            self.args.im_size,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
        )
        model.train()
        for epochs in range(self.args.epochs):
            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if step % 20 == 0:
                    self.logger.info(
                        f"Epoch: {epochs}, Loss: {loss.item()} at step {step}/{len(train_loader)}"
                    )
        return model

    def evaluate_model(self, model, device_id=0):
        """
        评估模型在测试集上的损失。
        """
        test_loader = torch.utils.data.DataLoader(
            self.dst_test, batch_size=self.args.batch, shuffle=False
        )
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
        return total_loss / len(self.dst_test)

    def select(self, **kwargs):
        """
        子类需实现具体的选择方法。
        """
        raise NotImplementedError("Subclasses should implement this method.")
        # 使用多个GPU进行并行计算
        # 示例：分配任务到 self.num_gpus 个GPU
        # ...existing code...
