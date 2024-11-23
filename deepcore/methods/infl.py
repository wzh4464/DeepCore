###
# File: ./deepcore/methods/infl.py
# Created Date: Saturday, November 23rd 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 23rd November 2024 5:00:35 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

from typing import override

from deepcore.methods.selection_methods import SELECTION_METHODS
from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel


class InflICML(EarlyTrain):
    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        repeat=10,
        specific_model=None,
        balance=False,
        **kwargs
    ):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self.balance = balance

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print(
                "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f"
                % (
                    epoch,
                    self.epochs,
                    batch_idx + 1,
                    (self.n_train // batch_size) + 1,
                    loss.item(),
                )
            )

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def finish_run(self):
        self.model.eval()

        # Gradient computation for influence calculation
        embedding_dim = self.model.get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers,
        )
        sample_num = self.n_train

        # Storage for influence scores
        self.influence_scores = torch.zeros(self.n_train, device=self.args.device)

        for i, (input, targets, _) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(
                outputs.requires_grad_(True), targets.to(self.args.device)
            ).sum()
            batch_num = targets.shape[0]

            # Compute gradients w.r.t. parameters
            grad_params = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )

            # Compute influence for each sample
            for idx in range(batch_num):
                per_sample_grad = [
                    g[idx].view(1, -1) for g in grad_params if g is not None
                ]  # Gradients for the current sample
                self.influence_scores[i * self.args.selection_batch + idx] = sum(
                    (p * p).sum() for p in per_sample_grad
                )

        self.model.train()

    def select(self, **kwargs):
        # Normalize influence scores
        self.influence_scores /= torch.max(self.influence_scores)

        if not self.balance:
            top_examples = self.train_indx[
                torch.argsort(self.influence_scores, descending=True)[
                    : self.coreset_size
                ]
            ]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(
                    top_examples,
                    c_indx[np.argsort(self.influence_scores[c_indx])[::-1][:budget]],
                )

        return {"indices": top_examples, "scores": self.influence_scores.cpu().numpy()}

    @override
    def get_scores(self):
        self.influence_scores = torch.zeros(self.n_train, device=self.args.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        return self.influence_scores.cpu().detach().numpy()

SELECTION_METHODS["infl"] = InflICML
