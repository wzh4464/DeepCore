from typing import override
from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel


class GraNd(EarlyTrain):

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        repeat=1,
        specific_model=None,
        balance=False,
        **kwargs,
    ):
        super().__init__(
            dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs
        )
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self.balance = balance

        # 使用基类方法获取特殊索引
        self.flipped_indices = self.get_special_indices("flipped")
        self.scores_indices = self.get_special_indices("selection")

        if self.flipped_indices:
            print(f"[GraNd] 跟踪{len(self.flipped_indices)}个翻转样本")
        if self.scores_indices:
            print(f"[GraNd] 为{len(self.scores_indices)}个样本计算分数")

    @override
    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        super().while_update(outputs, loss, targets, epoch, batch_idx, batch_size)
        if batch_idx % self.args.print_freq == 0:
            print(
                "| Epoch [%3d/%3d] Iter[%3d/%3d]\\t\\tLoss: %.4f"
                % (
                    epoch,
                    self.epochs,
                    batch_idx + 1,
                    (self.n_train // batch_size) + 1,
                    loss.item(),
                )
            )

    @override
    def before_run(self):
        super().before_run()
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    @override
    def after_epoch(self):
        super().after_epoch()

    @override
    def train(self, epoch, list_of_train_idx):
        return super().train(epoch, list_of_train_idx)

    @override
    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        super().after_loss(outputs, loss, targets, batch_inds, epoch)

    @override
    def finish_run(self):
        self.model.embedding_recorder.record_embedding = True
        self.model.eval()

        embedding_dim = self.model.get_last_layer().in_features
        train_loader, train_indices = self._get_train_loader()
        self.logger.info(f"Created training data loader")
        self.train_iterator = iter(train_loader)
        sample_num = self.n_train

        # 初始化所有样本的norm为NaN
        self.norm_matrix.fill_(float("nan"))

        if self.scores_indices:
            print(f"[GraNd] 只为{len(self.scores_indices)}个特定样本计算分数")

        for i, (input, targets, true_idx) in enumerate(train_loader):
            batch_indices = true_idx.numpy() if hasattr(true_idx, "numpy") else true_idx

            # 只保留需要 selection 的样本
            if self.scores_indices:
                mask = [idx in self.scores_indices for idx in batch_indices]
                if not any(mask):
                    continue  # 该 batch 没有 selection 样本，直接跳过
                # 转为 numpy 数组方便索引
                mask = np.array(mask)
                input = input[mask]
                targets = targets[mask]
                if hasattr(batch_indices, "shape"):
                    batch_indices = batch_indices[mask]
                else:
                    batch_indices = np.array(batch_indices)[mask]

            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(
                outputs.requires_grad_(True), targets.to(self.args.device)
            )

            # 计算每个样本的损失
            # 处理 loss 可能是标量的情况（batch 只有一个样本）
            if loss.dim() == 0:
                individual_losses = [loss]  # 将标量 loss 放入列表中
            else:
                individual_losses = torch.unbind(loss)

            for j in range(len(individual_losses)):
                sample_idx = (
                    batch_indices[j].item()
                    if hasattr(batch_indices[j], "item")
                    else batch_indices[j]
                )
                # 只为特定样本计算分数
                # if self.scores_indices and sample_idx not in self.scores_indices:
                #     continue
                matrix_idx = sample_idx

                # 计算单个样本的梯度
                self.model_optimizer.zero_grad()
                individual_losses[j].backward(retain_graph=True)

                # 计算所有参数梯度的平方和的平方根 (L2范数)
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm(2).item() ** 2
                grad_norm = grad_norm**0.5

                # 存储梯度范数
                self.norm_matrix[matrix_idx, self.cur_repeat] = grad_norm

        self.model.train()
        self.model.embedding_recorder.record_embedding = False

    @override
    def _initialize_data_loader(self):
        super()._initialize_data_loader()

    @override
    def select(self, **kwargs):
        self._initialize_data_loader()
        # Initialize a matrix to save norms of each sample on idependent runs
        window_size = kwargs.get("window_size", self.coreset_size)
        window_offset = kwargs.get("window_offset", 0.0)
        self.get_scores()

        # 使用滑动窗口选择样本
        def select_with_window(indices, scores, window_size, offset=0.0):
            """
            使用滑动窗口选择样本，可以排除得分最高的一些样本

            params:
                indices: 样本索引
                scores: 对应的分数
                window_size: 窗口大小（要选择的样本数量）
                offset: 窗口偏移，0表示从最高分开始，0.1表示排除前10%最高分的样本
            """
            # 按分数降序排序
            sorted_idx = np.argsort(scores)[::-1]
            total_samples = len(sorted_idx)

            # 计算窗口起始位置
            start_idx = int(offset * total_samples)
            # 确保不超出范围
            if start_idx + window_size > total_samples:
                end_idx = total_samples
            else:
                end_idx = start_idx + window_size

            return indices[sorted_idx[start_idx:end_idx]]

        if not self.balance:
            # 使用滑动窗口选择样本
            top_examples = select_with_window(
                self.train_indx, self.norm_mean, window_size, window_offset
            )
        else:
            # 平衡采样，每个类别使用滑动窗口选择样本
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                c_scores = self.norm_mean[c_indx]
                budget = round(self.fraction * len(c_indx))
                selected = select_with_window(c_indx, c_scores, budget, window_offset)
                top_examples = np.append(top_examples, selected)

        return {"indices": top_examples, "scores": self.norm_mean}

    @override
    def get_scores(self):
        # 获取scores_indices（如果可用）
        self.scores_indices = (
            self.dst_train.get_flipped_selection_from()
            if hasattr(self.dst_train, "get_flipped_selection_from")
            else []
        )
        if self.scores_indices:
            print(f"[GraNd] 将为{len(self.scores_indices)}个特定样本计算分数")
        # 初始化为NaN
        self.norm_matrix = torch.full(
            [self.n_train, self.repeat], float("nan"), requires_grad=False
        ).to(self.args.device)
        for self.cur_repeat in range(self.repeat):
            self.before_run()
            self.run()
        # 计算平均norm（忽略NaN值）
        self.norm_mean = torch.nanmean(self.norm_matrix, dim=1).cpu().detach().numpy()
        return self.norm_mean

    @override
    def run(self):
        self.random_seed = self.random_seed + self.cur_repeat
        super().run()
        self.random_seed = self.random_seed - self.cur_repeat
