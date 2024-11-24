###
# File: ./deepcore/methods/oti.py
# Created Date: Friday, August 9th 2024
# Author: Zihan
# -----
# Last Modified: Sunday, 24th November 2024 12:21:54 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
# 2024-08-17        Zihan	Added epoch usage tracking and removed manual memory clearing
# 2024-08-24        Zihan	Added support for multiple GPUs and improved code readability
# -----
# --dataset MNIST --model LeNet --selection OTI --num_exp 1 --epochs 5 --selection_epochs 5 --data_path ./data --gpu 0 1 2 --optimizer SGD --lr 0.1 --scheduler CosineAnnealingLR --save_path ./results --num_gpus 3 --oti_mode full
# Prec@1 93.200000
###

import os
import traceback
from .selection_methods import SELECTION_METHODS
from .earlytrain import EarlyTrain
import torch
import numpy as np
import torch.multiprocessing as mp
from torch import autograd
from functools import partial
from functorch import vmap, grad
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple, override
import pandas as pd
from torch.utils.data import DataLoader
from deepcore.datasets.flipped_dataset import IndexedDataset

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import ScoreTracker


class TqdmLoggingHandler(logging.Handler):
    """自定义日志处理器，使用 tqdm.write 以兼容 tqdm 进度条。"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class OTI(EarlyTrain):
    """
    Implements the Online Training Influence (OTI) method.

    This class saves model parameters after updating each data point, then uses these parameters
    to calculate a score for each data point. The final selection of the subset is based on these scores.
    """

    # Constructor and Initialization Methods
    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        mode="full",
        fractions=None,
        use_regularization=False,  # 新选项：是否使用正则化
        use_learning_rate=True,  # 新选项：是否使用学习率
        use_sliding_window=False,  # 新选项：是否使用滑动窗口（暂未实现）
        **kwargs,
    ):
        """
        Initialize the OTI method with training dataset and relevant parameters.

        Args:
            dst_train (Dataset): The training dataset.
            args (argparse.Namespace): Arguments containing various settings.
            fraction (float, optional): Fraction of the dataset to select as coreset. Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 200.
            specific_model (str, optional): Name of a specific model to use. Defaults to None.
        """
        if fractions is None:
            fractions = [0.8, 0.5, 0.3]
        super().__init__(
            dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs
        )

        # Force batch size to 1 for OTI method
        # self.args.selection_batch = 1

        self.current_epoch = 0
        self.current_step = 0
        self.total_params_processed = 0  # Total number of parameters processed
        self.epoch_data_orders = {}  # Store the data order for each epoch
        self.current_epoch_parameters = []  # Store parameters for the current epoch
        self.best_params = None  # To store the best parameters
        self.best_loss = float("inf")  # To store the best loss
        self.epoch_losses = []  # Track losses for each epoch
        self.epoch_usage = []  # Track whether each epoch was actually used
        self.initial_params = None  # To store initial parameters
        self.num_gpus = args.num_gpus if hasattr(args, "num_gpus") else 1
        self.mode = args.oti_mode if hasattr(args, "oti_mode") else mode
        self.fractions = fractions
        self.pseudo_params_list = []  # To store pseudo parameters for each data point
        self.lr_history = {}  # To store learning rates for each epoch
        self.use_regularization = use_regularization
        self.use_learning_rate = use_learning_rate
        self.use_sliding_window = use_sliding_window

        # Initialize score tracker
        self.score_tracker = ScoreTracker(len(dst_train), args.save_path)

        # Store initial seed
        self.initial_seed = random_seed if random_seed is not None else args.seed

        self.current_grads = None

        # Add tracking for flipped samples
        self.flipped_indices = (
            dst_train.get_flipped_indices()
            if hasattr(dst_train, "get_flipped_indices")
            else []
        )
        self.scores_indices = (
            dst_train.get_flipped_selection_from()
            if hasattr(dst_train, "get_flipped_selection_from")
            else []
        )

        if self.flipped_indices:
            self.logger.info(
                f"[OTI] Tracking {len(self.flipped_indices)} flipped samples"
            )
            self.logger.info(
                f"[OTI] Computing scores for {len(self.scores_indices)} samples"
            )

    def _update_seed(self):
        """Update seed based on current experiment number."""
        new_seed = self.initial_seed + self.score_tracker.current_exp
        self.args.seed = new_seed
        self.logger.info(
            f"[OTI] Updated seed to {new_seed} for experiment {self.score_tracker.current_exp + 1}"
        )
        return new_seed

    @override
    def before_run(self):
        """
        Perform pre-run setup tasks.

        1. super().before_run()
        2. Save initial parameters to file

        from super().before_run():
            Defined:
                model: The model to train.
                criterion: The loss function.
                model_optimizer: The optimizer for the model. (from super.setup_optimizer_and_scheduler())
                scheduler: The learning rate scheduler. (from super.setup_optimizer_and_scheduler())
        """
        super().before_run()
        # Save initial parameters
        self.initial_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }
        # Save initial parameters to file
        torch.save(
            self.initial_params,
            os.path.join(self.args.save_path, "initial_params.pt"),
        )
        self.logger.info(
            f"[OTI] Initial parameters saved to {self.args.save_path}/initial_params.pt"
        )

    @override
    def after_epoch(self):
        super().after_epoch()

        current_lr = self.get_lr()
        self.lr_history[self.current_epoch] = current_lr
        self.logger.info(
            f"[OTI] Epoch {self.current_epoch} finished. New LR: {current_lr}"
        )

        if self.scheduler:
            self.scheduler.step()  # 确保在 optimizer.step() 之后调用

        file_path = os.path.join(
            self.args.save_path, f"epoch_{self.current_epoch}_data.pkl"
        )

        with open(file_path, "wb") as f:
            torch.save(
                {
                    "parameters": self.current_epoch_parameters,
                    "data_order": self.epoch_data_orders[self.current_epoch],
                    "epoch_usage": (
                        self.epoch_losses[-1] < self.epoch_losses[-2]
                        if len(self.epoch_losses) > 1
                        else True
                    ),
                    "learning_rate": current_lr,  # 只保存当前epoch的学习率
                },
                f,
            )

        self.logger.info(
            f"[OTI] Parameters, data order, and learning rate saved for epoch {self.current_epoch}"
        )

        self.current_epoch_parameters = []
        self.current_step = 0
        self.current_epoch += 1

    @override
    def finish_run(self):
        """
        Finish the run by saving the best parameters and preserving all intermediate results.

        This method saves the best parameters to a separate file and keeps all the intermediate
        epoch data files for further analysis or debugging purposes.
        """
        if self.best_params is None:
            self.logger.warning(
                "[OTI] Warning: No best parameters were saved during the run."
            )
            self.best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
            self._save_best_params()
            self.logger.info(
                "[OTI] Run finished. All intermediate results have been preserved."
            )
        else:

            self.logger.info(
                "[OTI] Best parameters were successfully saved during the run."
            )

    # Training and Loss Handling Methods
    @override
    def train(self, epoch, list_of_train_idx):
        """
        Get the train index for each epoch.
        Insert a hook to store the data order for each epoch.
        """
        self.epoch_data_orders[epoch] = (
            list_of_train_idx.copy()
        )  # Store the data order for this epoch
        return super().train(epoch, list_of_train_idx)

    @override
    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        """Update best parameters after each epoch"""
        super().after_loss(outputs, loss, targets, batch_inds, epoch)

        current_loss = loss.mean().item()
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
            self._save_best_params()

        # Print progress at appropriate intervals
        if self.current_step % self.args.print_freq == 0:
            self.logger.info(
                f"|Training First Round| Epoch [{epoch}/{self.epochs}] Step [{self.current_step}/{(self.n_train // self.args.selection_batch)+1}]\t\tLoss: {current_loss:.4f}"
            )

        self.total_params_processed += len(batch_inds)
        self.current_step += 1

    @override
    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        """
        Perform minimal logging during the update step.
        All parameter saving is handled in after_loss.
        """
        super().while_update(outputs, loss, targets, epoch, batch_idx, batch_size)

        # Only handle progress logging
        if batch_idx % self.args.print_freq == 0:
            self.logger.debug(
                f"|Training First Round| Epoch [{epoch}/{self.epochs}] Iter[{batch_idx+1}/{(self.n_train // batch_size)+1}]\t\tLoss: {loss.item():.4f}"
            )

    def _save_best_params(self):
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        with open(best_params_path, "wb") as f:
            torch.save(self.best_params, f)
        # self.logger.info(f"[OTI] Best parameters saved to {best_params_path}")

    # Parameter Retrieval Methods

    def _load_best_params(self):
        """
        Load the best parameters from a file.

        Returns:
            dict: The best parameters loaded from the file.

        Raises:
            FileNotFoundError: If the best parameters file is not found.
        """
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        if not os.path.exists(best_params_path):
            raise FileNotFoundError(
                f"[OTI] Best parameters file not found at {best_params_path}"
            )
        with open(best_params_path, "rb") as f:
            return torch.load(f, weights_only=True)

    # Score Calculation Methods
    def _calculate_scores(
        self, use_regularization=False, use_learning_rate=True, use_sliding_window=False
    ):
        """
        Calculate scores by training in real-time and comparing parameters with best_params.
        """
        try:
            if self.best_params is None:
                self.before_run()
                self.best_params = self._load_best_params()
            self.logger.debug(
                "[OTI] Best parameters found. Starting score calculation."
            )
        except FileNotFoundError:
            self.before_run()
            self.run()

        init_params = torch.load(
            os.path.join(self.args.save_path, "initial_params.pt"), weights_only=True
        )

        if self.num_gpus <= 1:
            self.logger.info("[OTI] Using single GPU for score calculation")
            device_id = 0 if self.args.gpu is None else self.args.gpu[0]
            return self._single_gpu_calculate_scores(
                self.best_params,
                init_params,
                device_id,
                use_regularization,
                use_learning_rate,
                # use_sliding_window
            )
        else:
            self.logger.info("[OTI] Using multiple GPUs for score calculation")
            return self._multi_gpu_calculate_scores(
                self.best_params,
                init_params,
                use_regularization,
                use_learning_rate,
                # use_sliding_window
            )

    def _calculate_l2_distance(self, params1, params2, device):
        """Calculate L2 distance between two parameter sets"""
        return sum(
            torch.norm(params1[name].to(device) - params2[name].to(device)).item()
            for name in params1
            if name in params2
        )

    def _calculate_pseudo_params(self, params, grads, learning_rate):
        """Calculate pseudo parameters for a given set of parameters and gradients."""
        return {
            name: params[name] - learning_rate * grads[name]
            for name, grad in grads.items()
            if grad is not None
        }

    def _get_train_loader(self):
        """Create and return training data loader."""
        self.logger.info("Creating training data loader.")
        # Create a list of indices for training
        # list_of_train_idx = np.random.choice(
        #     np.arange(self.n_train), self.n_pretrain_size, replace=False
        # )

        # Create the indexed dataset
        indexed_dataset = self.dst_train
        self.logger.debug(
            "IndexedDataset created with %d samples.", len(indexed_dataset)
        )

        # Create DataLoader
        train_loader = torch.utils.data.DataLoader(
            indexed_dataset,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )
        self.logger.info(
            "Training data loader created with batch size %d and %d workers.",
            self.args.selection_batch,
            self.args.workers,
        )
        return train_loader, self.dst_train.indices

    def _init_multiprocessing(self):
        """Initialize multiprocessing manager and return dict."""
        from contextlib import suppress

        with suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)
        return mp.Manager().dict()

    def _start_worker_processes(
        self,
        device_id,
        epochs_per_worker,
        best_params,
        use_regularization,
        use_learning_rate,
        return_dict,
        worker_offset=0,
    ):
        """Start worker processes for a specific device."""
        processes = []
        for worker_id, epochs in enumerate(epochs_per_worker):
            if not epochs:  # Skip if no work to do
                continue

            actual_worker_id = worker_offset + worker_id
            try:
                self.logger.info(
                    f"Starting worker {actual_worker_id} with {len(epochs)} epochs"
                )
                p = mp.Process(
                    target=self._worker_process_wrapper,
                    args=(
                        device_id,
                        epochs,
                        best_params,
                        use_regularization,
                        use_learning_rate,
                        return_dict,
                        actual_worker_id,
                    ),
                )
                # 移除 daemon=True 设置
                processes.append(p)
                p.start()
                self.logger.info(f"Worker {actual_worker_id} started successfully")
            except Exception as e:
                self.logger.error(
                    f"Failed to start worker {actual_worker_id}: {str(e)}"
                )

        return processes

    def _single_gpu_calculate_scores(
        self,
        best_params,
        init_params,
        device_id,
        use_regularization=False,
        use_learning_rate=True,
        cpus_per_gpu=4,
    ) -> torch.Tensor:
        """Calculate scores using single GPU with multiple workers."""
        self.logger.info("[OTI] Starting score calculation on single GPU")
        return_dict = self._init_multiprocessing()

        # Load the model with initial parameters
        self.model.load_state_dict(init_params)
        self.logger.info("[OTI] Loaded initial parameters")

        scores = []
        for name, param in best_params.items():
            best_params[name] = param.to(device_id)

        # Calculate scores
        for epoch in range(self.epochs):
            scores.append(
                self._calculate_scores_on_device(
                    device_id,
                    [epoch],
                    best_params,
                    use_regularization,
                    use_learning_rate,
                    return_dict,
                )
            )

        # save scores to csv, and then sum them up and to torch tensor and return
        scores_df = pd.DataFrame({
            'epoch': range(self.epochs),
            'scores': [s.mean().item() for s in scores]
        })
        self._save_epoch_scores_to_csv(
            "epoch_scores.csv", scores_df, '[OTI] Saved epoch scores to '
        )
        # Stack scores and compute mean across epochs
        stacked_scores = torch.stack(scores)
        final_scores = torch.mean(stacked_scores, dim=0)

        return final_scores

    def _multi_gpu_calculate_scores(
        self,
        best_params,
        use_regularization=False,
        use_learning_rate=True,
        cpus_per_gpu=4,
    ):
        """Calculate scores using multiple GPUs with multiple workers per GPU."""
        raise NotImplementedError("Multi-GPU support is not implemented yet")

    def _calculate_scores_on_device(
        self,
        device_id: int,
        epochs_to_process: list,
        best_params: dict,
        use_regularization: bool = False,
        use_learning_rate: bool = True,
        return_dict: Optional[Dict] = None,
        worker_id: Optional[int] = None,
        train_loader: Optional[DataLoader] = None,
        train_indices: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        try:
            worker_name = (
                f"Worker-{worker_id}" if worker_id is not None else f"GPU-{device_id}"
            )
            device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

            self._setup_optimizer_scheduler(use_learning_rate)
            self.logger.info(f"[{worker_name}] Initialized optimizer and scheduler")

            if train_loader is None:
                train_loader, train_indices = self._get_train_loader()
                self.logger.info(f"[{worker_name}] Created training data loader")
                self.train_iterator = iter(train_loader)

            scores = torch.zeros(len(train_indices), dtype=torch.float32, device="cpu")

            for epoch in epochs_to_process:
                epoch_lr = self._update_learning_rate(use_learning_rate, epoch)
                self.logger.info(f"[{worker_name}] Epoch {epoch} using lr: {epoch_lr}")

                for batch_ind, (inputs, targets, true_idx) in enumerate(
                    self.train_iterator
                ):
                    inputs, targets = inputs.to(device), targets.to(device)
                    batch_scores, batch_indices = self._process_batch(
                        inputs,
                        targets,
                        batch_ind,
                        best_params,
                        epoch_lr,
                        device,
                        use_regularization,
                        worker_name,
                        epoch,
                        train_indices,
                        true_idx,
                    )
                    scores[batch_indices] = batch_scores.cpu()

            if return_dict is not None:
                # 修改此行，断开梯度追踪并确保张量在CPU上
                return_dict[worker_id if worker_id is not None else device_id] = (
                    scores.detach().cpu()
                )
                
            scores = scores[self.scores_indices] if self.scores_indices else scores

            return scores

        except Exception as e:
            self.logger.error(f"[{worker_name}] Error: {str(e)}")
            self.logger.error(f"[{worker_name}] Traceback: {traceback.format_exc()}")
            raise e
            # if return_dict is not None:
            #     return_dict[worker_id if worker_id is not None else device_id] = (
            #         torch.zeros(len(train_indices))
            #     )
            # return torch.zeros(len(train_indices))

    def _setup_optimizer_scheduler(self, use_learning_rate: bool):
        if use_learning_rate:
            self.model_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
            if self.args.scheduler == "CosineAnnealingLR":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model_optimizer, T_max=self.args.selection_epochs
                )

    def _update_learning_rate(self, use_learning_rate: bool, epoch: int) -> float:
        # 移除循环，避免在优化器步骤之前调用 scheduler.step()
        if use_learning_rate and self.scheduler:
            # 仅返回当前的学习率
            return self.model_optimizer.param_groups[0]["lr"]
        return 1.0

    def _process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_idx: int,
        best_params: dict,
        epoch_lr: float,
        device: torch.device,
        use_regularization: bool,
        worker_name: str,
        epoch: int,
        train_indices: np.ndarray,
        true_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a batch of data during training.
        Args:
            inputs (torch.Tensor): The input data for the batch.
            targets (torch.Tensor): The target labels for the batch.
            batch_idx (int): The index of the current batch.
            best_params (dict): The best parameters found so far.
            epoch_lr (float): The learning rate for the current epoch.
            device (torch.device): The device to run the computations on.
            use_regularization (bool): Whether to use regularization.
            worker_name (str): The name of the worker processing the batch.
            epoch (int): The current epoch number.
            train_indices (np.ndarray): The indices of the training data.
            true_idx (true_idx): The true index of the batch
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the scores and the batch indices tensor.
        """

        batch_start = batch_idx * self.args.selection_batch
        batch_end = min((batch_idx + 1) * self.args.selection_batch, len(train_indices))
        batch_indices = train_indices[batch_start:batch_end]
        batch_indices_tensor = torch.tensor(batch_indices, device=device)

        # 计算整个batch的loss和梯度
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.model_optimizer.zero_grad()
        loss.backward()

        # push parameters and gradients (in case of change from _compute_scores)
        self.current_epoch_parameters.append(
            {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
        )

        self.current_grads = {
            name: param.grad.cpu().clone().detach()
            for name, param in self.model.named_parameters()
        }

        scores = self._compute_scores(
            best_params, epoch_lr, use_regularization, device, inputs, targets, true_idx
        )

        # pop parameters and gradients and load to model
        for name, param in self.model.named_parameters():
            param.grad = self.current_grads[name].to(device)
        self.model.load_state_dict(self.current_epoch_parameters[-1])

        self.model_optimizer.step()

        if batch_idx % 20 == 0:
            self.logger.info(
                f"[{worker_name}] Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}, "
                f"Samples scored: {len(scores)}, "
                f"Mean score: {scores.mean().item():.4f}"
            )
        return scores, batch_indices_tensor

    def _compute_scores(
        self,
        best_params: dict,
        epoch_lr: float,
        use_regularization: bool,
        device: torch.device,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        true_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for a batch of samples, with special handling for flipped samples.
        """
        batch_size = inputs.size(0)
        initial_distances = torch.zeros(batch_size, device=device)
        pseudo_distances = torch.zeros(batch_size, device=device)

        # Calculate initial distances
        for name, param in self.model.named_parameters():
            if name in best_params:
                param_diff = param - best_params[name]
                initial_distances += torch.norm(param_diff.view(1, -1), dim=1)

        # Process each sample individually
        for i in range(batch_size):
            true_idx_i = true_idx[i].item()

            # Skip if sample is not in scores_indices
            if self.scores_indices and true_idx_i not in self.scores_indices:
                continue

            # Clear previous gradients
            self.model_optimizer.zero_grad()
            if i % self.args.print_freq == 0:
                self.logger.debug(
                    f"Processing sample {i}/{batch_size} for score calculation"
                )

            # 当前样本的前向传播
            input_i = inputs[i : i + 1]  # 提取单个样本，形状 (1, ...)
            target_i = targets[i : i + 1]  # 提取对应标签，形状 (1, ...)
            output_i = self.model(input_i)
            loss_i = self.criterion(output_i, target_i)

            # Backward pass
            loss_i.backward(retain_graph=True)

            # Calculate pseudo-parameters and distances
            for name, param in self.model.named_parameters():
                if name in best_params and param.grad is not None:
                    grad_i = param.grad

                    # Flatten tensors for distance calculation
                    param_flat = param.view(1, -1)
                    grad_flat = grad_i.view(1, -1)
                    best_param_flat = best_params[name].view(1, -1)

                    # Calculate pseudo-parameter and its distance to best parameter
                    pseudo_param = param_flat - epoch_lr * grad_flat
                    param_diff_pseudo = pseudo_param - best_param_flat

                    # Update pseudo-distances
                    pseudo_distances[i] += torch.norm(param_diff_pseudo, dim=1).item()

        # Calculate final scores
        scores = torch.where(
            initial_distances > 0,
            (
                (initial_distances - pseudo_distances)
                / (initial_distances + pseudo_distances)
                if use_regularization
                else initial_distances - pseudo_distances
            ),
            torch.zeros_like(initial_distances),
        )

        return scores

    def _load_scores(self):
        """Load pre-computed scores from file"""
        scores_path = os.path.join(self.args.save_path, "oti_scores.pkl")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Pre-computed scores not found at {scores_path}")
        with open(scores_path, "rb") as f:
            return torch.load(f, weights_only=True)

    # main method
    @override
    def select(
        self,
        use_regularization=False,
        use_learning_rate=True,
        use_sliding_window=False,
        **kwargs,
    ):
        """
        Select the subset based on calculated scores.

        Returns:
            dict: A dictionary containing the selected indices and their scores.
        """

        # Initialize data loader
        self._initialize_data_loader()

        # Calculate scores
        score_array, indices = self._get_score(
            use_regularization, use_learning_rate, use_sliding_window
        )

        # Create DataFrame with scores
        df, top_k, selected_indices = self._select_top_k_scores(score_array, indices)

        # Save selected indices and their scores
        self._save_selected_scores(score_array, df, top_k, selected_indices)

        return {"indices": selected_indices, "scores": score_array}

    def _save_selected_scores(self, score_array, df, top_k, selected_indices):
        """
        Save the selected scores to a CSV file and log the details.
        Parameters:
        score_array (numpy.ndarray): Array of scores from which the top_k scores are selected.
        df (pandas.DataFrame): DataFrame containing the data with an "index" column.
        top_k (int): Number of top scores to select.
        selected_indices (list or numpy.ndarray): Indices of the selected top_k scores.
        Returns:
        None
        """

        selected_df = df[df["index"].isin(selected_indices)]
        self._save_epoch_scores_to_csv(
            "oti_selected_scores.csv",
            selected_df,
            '[OTI] Saved selected scores to ',
        )
        self.logger.info(f"[OTI] Selected {top_k} samples based on scores.")
        self.logger.info(f"[OTI] Selected scores: {score_array[selected_indices]}")

    def _select_top_k_scores(self, score_array, indices):
        """
        Select top-k scores with priority for flipped samples.
        """
        df = pd.DataFrame({"index": indices, "score": score_array})

        # Mark flipped samples in DataFrame
        df["is_flipped"] = df["index"].isin(self.flipped_indices)

        # Sort by score in descending order
        df = df.sort_values("score", ascending=False)

        self._save_epoch_scores_to_csv(
            "oti_scores.csv", df, '[OTI] Saved scores to '
        )
        # Select top-k samples
        top_k = self.coreset_size
        selected_indices = indices[np.argsort(score_array)[::-1][:top_k]]

        # Log flipped samples detection
        if self.flipped_indices:
            detected_flipped = set(selected_indices) & set(self.flipped_indices)
            self.logger.info(
                f"[OTI] Detected {len(detected_flipped)} out of {len(self.flipped_indices)} flipped samples"
            )

        return df, top_k, selected_indices

    # TODO Rename this here and in `_single_gpu_calculate_scores`, `_save_selected_scores` and `_select_top_k_scores`
    def _save_epoch_scores_to_csv(self, arg0, arg1, arg2):
        scores_path = os.path.join(self.args.save_path, arg0)
        arg1.to_csv(scores_path, index=False)
        self.logger.info(f"{arg2}{scores_path}")

    def _get_score(self, use_regularization, use_learning_rate, use_sliding_window):
        """
        Computes and returns the scores based on the specified mode.
        Parameters:
        use_regularization (bool): Flag to indicate whether to use regularization in score calculation.
        use_learning_rate (bool): Flag to indicate whether to use learning rate in score calculation.
        use_sliding_window (bool): Flag to indicate whether to use sliding window in score calculation.
        Returns:
        tuple: A tuple containing:
            - score_array (numpy.ndarray): The computed scores as a numpy array.
            - indices (numpy.ndarray): The indices of the training data as a numpy array.
        Raises:
        ValueError: If the mode is invalid or if the best parameters are not available.
        FileNotFoundError: If the stored parameters file is not found in "stored" mode.
        """

        if self.mode == "full":
            self.before_run()
            self.run()  # Run the training process
            if self.best_params is None:
                self.logger.error(
                    "self.best_params is None - model has not been trained yet"
                )
                raise ValueError(
                    "self.best_params is None - model has not been trained yet"
                )
            scores = self._calculate_scores(
                use_regularization, use_learning_rate, use_sliding_window
            )
        elif self.mode == "stored":
            try:
                self.before_run()
                self.best_params = self._load_best_params()
                if self.best_params is None:
                    self.logger.error("Failed to load best parameters from stored data")
                    raise ValueError("Failed to load best parameters from stored data")
                scores = self._calculate_scores(
                    use_regularization, use_learning_rate, use_sliding_window
                )
            except FileNotFoundError as e:
                self.logger.error(f"Error loading stored data: {str(e)}")
                raise
        elif self.mode == "scores":
            scores = self._load_scores()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Convert scores to numpy array
        score_array = scores.detach().cpu().numpy()
        indices = torch.arange(self.n_train).cpu().numpy()
        return score_array, indices

    def _get_current_batch(self):
        """Get current batch of data with indices."""
        self.logger.debug("Fetching current batch.")

        try:
            # Get next batch from iterator
            inputs, targets, batch_indices = next(self.train_iterator)
            self.logger.debug("Fetched batch with indices: %s", batch_indices)
        except StopIteration:
            # If iterator is exhausted, create new one
            self.logger.info("Train iterator exhausted. Reinitializing iterator.")
            self.train_iterator = iter(self.train_loader)
            inputs, targets, batch_indices = next(self.train_iterator)
            self.logger.debug("Fetched batch with indices: %s", batch_indices)

        return inputs, targets, batch_indices

    def _initialize_data_loader(self):
        """
        Initializes the data loader for training.
        Make sure:
        - self.train_loader is initialized with the training data.
        - self.train_indices is initialized with the indices of the training data.
        - self.train_iterator is initialized with the iterator for the training loader.
        """

        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.logger.info("Train loader or iterator not found. Reinitializing.")
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

    @override
    def get_scores(self, **kwargs):
        """torch.Tensor: Get the calculated scores."""
        return self._calculate_scores(True, True, False)


# Add OTI to SELECTION_METHODS
SELECTION_METHODS["OTI"] = OTI
