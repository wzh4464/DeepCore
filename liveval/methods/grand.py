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
        **kwargs
    ):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self.balance = balance
        
        # 添加对flipped indices的跟踪
        self.flipped_indices = (
            dst_train.get_flipped_indices()
            if hasattr(dst_train, "get_flipped_indices")
            else []
        )
        # 添加对scores indices的跟踪
        self.scores_indices = (
            dst_train.get_flipped_selection_from()
            if hasattr(dst_train, "get_flipped_selection_from")
            else []
        )
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
        self.model.embedding_recorder.record_embedding = (
            True  # recording embedding vector
        )

        self.model.eval()

        embedding_dim = self.model.get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers,
        )
        sample_num = self.n_train

        for i, (input, targets, _) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(
                outputs.requires_grad_(True), targets.to(self.args.device)
            ).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                self.norm_matrix[
                    i
                    * self.args.selection_batch : min(
                        (i + 1) * self.args.selection_batch, sample_num
                    ),
                    self.cur_repeat,
                ] = torch.norm(
                    torch.cat(
                        [
                            bias_parameters_grads,
                            (
                                self.model.embedding_recorder.embedding.view(
                                    batch_num, 1, embedding_dim
                                ).repeat(1, self.args.num_classes, 1)
                                * bias_parameters_grads.view(
                                    batch_num, self.args.num_classes, 1
                                ).repeat(1, 1, embedding_dim)
                            ).view(batch_num, -1),
                        ],
                        dim=1,
                    ),
                    dim=1,
                    p=2,
                )

        self.model.train()

        self.model.embedding_recorder.record_embedding = False

    @override
    def _initialize_data_loader(self):
        super()._initialize_data_loader()

    @override
    def select(self, **kwargs):
        self._initialize_data_loader()
        # Initialize a matrix to save norms of each sample on idependent runs
        self.get_scores()
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][
                : self.coreset_size
            ]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(
                    top_examples,
                    c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]],
                )

        return {"indices": top_examples, "scores": self.norm_mean}

    @override
    def get_scores(self):
        # 获取scores_indices（如果可用）
        self.scores_indices = (
            self.dst_train.get_flipped_selection_from()
            if hasattr(self.dst_train, "get_flipped_selection_from")
            else []
        )
        # 如果有scores_indices，则打印信息
        if self.scores_indices:
            print(f"[GraNd] 为{len(self.scores_indices)}个特定样本计算分数")
        # 计算标准GraNd分数
        self.norm_matrix = torch.zeros(
            [self.n_train, self.repeat], requires_grad=False
        ).to(self.args.device)
        for self.cur_repeat in range(self.repeat):
            self.before_run()
            self.run()
            self.random_seed = self.random_seed + 5
        # 计算平均norm
        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        # 如果有scores_indices，则创建一个新的分数数组，只为特定的索引设置分数
        if self.scores_indices:
            # 创建全为nan的数组
            filtered_scores = np.full(len(self.dst_train.indices), np.nan)
            # 只为scores_indices中的索引填充实际分数
            for i, idx in enumerate(self.dst_train.indices):
                if idx in self.scores_indices:
                    filtered_scores[i] = self.norm_mean[i]
            return filtered_scores
        else:
            # 如果没有scores_indices，返回所有分数
            return self.norm_mean
