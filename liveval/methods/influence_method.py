###
# File: ./liveval/methods/influence_method.py
# Created Date: Thursday, November 21st 2024
# Author: Zihan
# -----
# Last Modified: Sunday, 11th May 2025 12:15:06 am
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
            for step, (inputs, targets, _) in enumerate(train_loader):
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
            for batch in test_loader:
                inputs = batch[0]
                targets = batch[1]
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

    def train_for_epochs(self, num_epochs, train_indices, test_loader=None):
        """
        训练模型指定轮数，并在每轮后可选地进行测试。

        参数：
            num_epochs (int): 训练的轮数。
            train_indices (list): 训练样本的索引列表。
            test_loader (DataLoader, optional): 测试数据加载器。默认为 None。
        返回：
            float: 如果提供 test_loader，返回训练后最终的测试准确率。
        """
        self.logger.info(f"Training for {num_epochs} epochs with {len(train_indices)} samples")

        # 确保模型已初始化
        if not hasattr(self, 'model') or self.model is None:
            if hasattr(self, 'before_run'):
                self.before_run()

        test_acc = None

        for epoch in range(num_epochs):
            if hasattr(self, 'before_epoch'):
                self.before_epoch()
            if hasattr(self, 'train'):
                self.train(epoch, train_indices)
            # 如有需要，进行测试
            if test_loader is not None and hasattr(self, 'test'):
                _ = self.test(epoch)
            if hasattr(self, 'after_epoch'):
                self.after_epoch()

        # 最终评估
        if test_loader is not None and hasattr(self, 'model'):
            self.model.eval()
            correct = 0
            total = 0
            import torch
            with torch.no_grad():
                for inputs, targets, *_ in test_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = correct / total if total > 0 else 0.0
            self.last_test_acc = test_acc
            self.logger.info(f"Final test accuracy: {test_acc:.4f}")
        return test_acc
