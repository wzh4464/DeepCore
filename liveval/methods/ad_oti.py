###
# File: ./liveval/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Friday, 23rd May 2025 10:08:00 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import torch
import os
import numpy as np
import psutil
from torch import autograd
from functools import partial
from functorch import vmap, grad
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, override
import pandas as pd
import csv

from .selection_methods import SELECTION_METHODS
from .oti import OTI
from liveval.datasets.flipped_dataset import IndexedDataset

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import ScoreTracker, custom_collate


class AD_OTI(OTI):
    """
    AD_OTI: Adaptive version of OTI with optimized training and parameter management.

    This class implements an adaptive version of the Online Training Incremental (OTI) method,
    which optimizes the selection of training data based on cumulative valuations and loss changes.
    It manages parameters and training steps efficiently to improve model performance.

    Args:
        dst_train: Training dataset.
        args: Configuration arguments.
        fraction: Fraction of data to select (default is 0.5).
        random_seed: Seed for random number generation (default is None).
        epochs: Number of training epochs (default is 200).
        specific_model: Name of a specific model to use (default is None).
        delta_0: Initial window size (default is 2).
        delta_min: Minimum window size (default is 1).
        delta_max: Maximum window size (default is 3).
        delta_step: Step size for window adjustment (default is 1).
        eps_min: Lower threshold for loss change (default is 0.1).
        eps_max: Upper threshold for loss change (default is 0.05).
    """

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        mode="full",
        delta_0=2,  # Initial window size
        delta_min=1,  # Minimum window size
        delta_max=3,  # Maximum window size
        delta_step=1,  # Window size adjustment step
        eps_max=0.1,  # Lower threshold for loss change
        eps_min=0.05,  # Upper threshold for loss change
        **kwargs,
    ):
        """
        Initialize AD_OTI with LiveVal parameters.

        Args:
            dst_train: Training dataset
            args: Configuration arguments
            fraction: Fraction of data to select
            delta_0: Initial window size
            delta_min: Minimum window size
            delta_max: Maximum window size
            delta_step: Step size for window adjustment
            eps_max: Minimum loss change threshold
            eps_min: Maximum loss change threshold
        """
        super().__init__(
            dst_train=dst_train,
            args=args,
            fraction=fraction,
            random_seed=random_seed,
            epochs=epochs,
            specific_model=specific_model,
            mode=mode,
            **kwargs,
        )

        # LiveVal hyperparameters
        self.delta = int(delta_0)
        self.delta_min = int(delta_min)
        self.delta_max = int(delta_max)
        self.delta_step = int(delta_step)
        self.eps_max = eps_max
        self.eps_min = eps_min

        # log
        self.logger.info(
            f"AD_OTI initialized with parameters: delta={self.delta}, delta_min={self.delta_min}, delta_max={self.delta_max}, delta_step={self.delta_step}, eps_max={self.eps_max}, eps_min={self.eps_min}"
        )

        # Initialize storage for parameters and valuations
        # Each entry in param_queue will be a tuple:
        # (t, params, learning_rate, batch_indices)
        self.param_queue = deque(
            maxlen=self.delta_max
        )  # FIFO queue for parameters and states
        self.valuation_queue = deque()  # Queue to manage (t, t') pairs
        self.valuations = defaultdict(float)  # Cumulative valuations

        # Initialize state storage
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0

        # Track detected flipped samples per epoch
        self.detected_flipped_per_epoch = []

        self.logger.info(
            "AD_OTI initialized with parameters: fraction=%s, epochs=%s",
            fraction,
            epochs,
        )

    @override
    def select(self, **kwargs) -> dict:
        """
        Selects a subset of training data based on cumulative valuations and loss changes.

        This method implements the LiveVal algorithm to evaluate and select data points that
        contribute most effectively to the training process. It dynamically adjusts the selection
        criteria based on the observed loss changes during training.

        Args:
            **kwargs: Additional keyword arguments for customization.

        Returns:
            dict: A dictionary containing selected indices, their cumulative valuations,
                and the fraction of data selected.
        """
        self.logger.info("Starting optimized selection process.")
        self.before_run()
        self._initialize_data_loader()

        T = len(self.train_loader) * self.epochs
        v_cumulative = torch.zeros(self.n_train, device=self.args.device)
        delta = self.delta
        Q_ref = deque()
        Q_theta = deque()
        L_prev = None
        # 优化：只存批次索引
        batch_indices_history = {}  # step: batch_indices
        loss_history = deque(maxlen=self.delta_max + 1)

        theta_prev_tensor, self.param_shapes, self.param_sizes = self._dict_to_tensor(
            self.model.state_dict()
        )
        Q_theta.append((0, theta_prev_tensor.to(self.args.device)))
        self.logger.info("Initial model parameters saved to queue.")

        for t in range(1, T + 1):
            # 异常处理，保证 iterator 不会因异常退出
            try:
                inputs, targets, batch_indices = self._get_current_batch()
            except Exception as e:
                self.logger.error(f"Error getting batch at step {t}: {e}")
                self.train_iterator = iter(self.train_loader)
                inputs, targets, batch_indices = next(self.train_iterator)
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)

            self._train_step(inputs, targets, t, T)

            theta_t_tensor, _, _ = self._dict_to_tensor(self.model.state_dict())
            Q_theta.append((t, theta_t_tensor.to(self.args.device)))

            while len(Q_theta) > 0 and Q_theta[0][0] < t - self.delta_max:
                Q_theta.popleft()

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs)
                L_t = self.criterion(outputs, targets).item()
            self.model.train()

            loss_history.append(L_t)
            if len(loss_history) >= delta:
                L_delta_ago = loss_history[-delta]
                dot_L = (L_t - L_delta_ago) / delta
            else:
                dot_L = 0.0
            delta, L_prev = self._adjust_window_size(t, L_prev, L_t, delta, dot_L)

            t_prime = min(t + delta - 1, T)
            Q_ref.append((t, t_prime))

            # 只存批次索引到CPU，节省内存
            if isinstance(batch_indices, torch.Tensor):
                batch_indices_history[t] = batch_indices.clone().cpu()
            else:
                batch_indices_history[t] = torch.tensor(batch_indices).clone().cpu()
            while batch_indices_history and min(batch_indices_history.keys()) < t - self.delta_max:
                del batch_indices_history[min(batch_indices_history.keys())]

            self._process_reference_pairs(
                v_cumulative, Q_ref, Q_theta, t, batch_indices_history
            )

            if self.flipped_indices and t % self.args.print_freq == 0:
                num_detected = self.count_flipped_in_lowest_scores(
                    self.logger, self.args, self.flipped_indices, v_cumulative
                )
                self.detected_flipped_per_epoch.append(
                    {"step": t, "detected": num_detected}
                )
                self.logger.info(
                    f"[AD_OTI] Step {t}: Detected {num_detected}/{len(self.flipped_indices)} flipped samples."
                )

        if self.flipped_indices:
            detection_df = pd.DataFrame(self.detected_flipped_per_epoch)
            detection_path = os.path.join(
                self.args.save_path, "ad_oti_detection_stats.csv"
            )
            detection_df.to_csv(detection_path, index=False)
            self.logger.info(
                f"Saved flipped sample detection statistics to {detection_path}"
            )

        return self._select_top_samples(v_cumulative)

    def _process_reference_pairs(self, v_cumulative, Q_ref, Q_theta, t, batch_indices_history):
        """
        处理参考对并更新累计估值，只使用批次索引历史。
        """
        while Q_ref and Q_ref[0][1] == t:
            t_1, t_2 = Q_ref.popleft()
            self.logger.debug("Processing reference pair (t1=%d, t2=%d).", t_1, t_2)
            theta_t2 = next((theta for step, theta in Q_theta if step == t_2), None)
            theta_t1_prev = next(
                (theta for step, theta in Q_theta if step == t_1 - 1), None
            )
            # 使用批次索引历史
            if (
                t_1 in batch_indices_history
                and theta_t2 is not None
                and theta_t1_prev is not None
            ):
                batch_indices_t1 = batch_indices_history[t_1]
                theta_t2 = theta_t2.to(self.args.device)
                theta_t1_prev = theta_t1_prev.to(self.args.device)
                # 传递时刻 t_1 用于重新获取数据
                self._update_cumulative_valuations_with_step(
                    v_cumulative, batch_indices_t1, theta_t2, theta_t1_prev, t_1
                )
            Q_theta = deque(
                [(step, theta) for step, theta in Q_theta if step >= t_1 - 1]
            )
            self.logger.info(
                "Cleaned up model parameters from queue up to step %d.", t_1 - 1
            )

    def _update_cumulative_valuations_with_step(
        self, v_cumulative, batch_indices, theta_t2, theta_t1_prev, step_t1
    ):
        """
        更新累计估值，通过批次索引重新获取数据。
        """
        # 将批次索引移到正确的设备
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.cpu().tolist()
        data_batch = []
        target_batch = []
        true_indices = []
        for idx in batch_indices:
            data, target, true_idx = self.dst_train[idx]
            data_batch.append(data)
            target_batch.append(target)
            true_indices.append(true_idx)
        data_batch = torch.stack(data_batch).to(self.args.device)
        target_batch = torch.tensor(target_batch).to(self.args.device)
        true_indices = torch.tensor(true_indices)
        gradients_tensor = self._batch_compute_gradients(
            theta_t1_prev, data_batch, target_batch
        )
        eta_t1 = self.get_lr()
        batch_pseudo_params_tensor = (
            theta_t1_prev.unsqueeze(0) - eta_t1 * gradients_tensor
        )
        v_i_t1 = self._batch_compute_valuations(
            theta_t2, theta_t1_prev, batch_pseudo_params_tensor
        )
        for i, true_idx in enumerate(true_indices):
            if self.scores_indices and true_idx.item() not in self.scores_indices:
                continue
            v_cumulative[true_idx] += v_i_t1[i]

    def _select_top_samples(self, v_cumulative):
        """
        Select top samples based on cumulative valuations, with support for flipped sample tracking.

        Args:
            v_cumulative (torch.Tensor): Tensor of cumulative valuations

        Returns:
            dict: Selection results with indices, valuations, and metadata
        """
        k = int(self.fraction * len(self.dst_train))

        # Move to CPU for numpy operations
        v_cpu = v_cumulative.cpu()

        # Create DataFrame with scores and flipped status
        df = pd.DataFrame(
            {"index": np.arange(len(self.dst_train)), "score": v_cpu.numpy()}
        )

        # Mark flipped samples if present
        if self.flipped_indices:
            df["is_flipped"] = df["index"].isin(self.flipped_indices)

        # Sort by score in descending order
        df = df.sort_values("score", ascending=False)

        # Save scores to CSV
        self._save_epoch_scores_to_csv(
            "ad_oti_scores.csv", df, "[AD_OTI] Saved scores to "
        )

        # Select top-k samples
        selected_indices = df["index"].iloc[:k].values

        # Log flipped sample detection if applicable
        if self.flipped_indices:
            detected_flipped = set(selected_indices) & set(self.flipped_indices)
            self.logger.info(
                f"[AD_OTI] Detected {len(detected_flipped)} out of {len(self.flipped_indices)} flipped samples"
            )

            # Save selected samples with scores
            selected_df = df[df["index"].isin(selected_indices)]
            self._save_epoch_scores_to_csv(
                "ad_oti_selected_scores.csv",
                selected_df,
                "[AD_OTI] Saved selected scores to ",
            )

        # Save results to file
        result = {
            "indices": selected_indices,
            "valuation": v_cumulative.cpu().numpy(),
            "fraction": self.fraction,
        }

        save_path = os.path.join(self.args.save_path, "selection_result.pt")
        torch.save(result, save_path, weights_only=False)
        self.logger.info(f"Saved selection results to {save_path}")
        self.logger.info("Selection process completed.")
        return result

    def _update_cumulative_valuations(
        self, v_cumulative, batch_indices, theta_t2, theta_t1_prev
    ):
        """
        Update cumulative valuations for each sample in the batch,
        with support for individual sample gradient computation.

        Args:
            v_cumulative (torch.Tensor): Tensor of cumulative valuations
            batch_indices (torch.Tensor): Indices of samples in the batch
            theta_t2 (torch.Tensor): Parameters at time t2
            theta_t1_prev (torch.Tensor): Parameters at time t1-1
        """
        # Handle batch_indices correctly - it might be a tensor or a list
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.cpu().tolist()

        # Use IndexedDataset to get samples by index
        data_batch = []
        target_batch = []
        true_indices = []

        for idx in batch_indices:
            # Get sample and add to batch
            data, target, true_idx = self.dst_train[idx]
            data_batch.append(data)
            target_batch.append(target)
            true_indices.append(true_idx)

        # Stack batch data
        data_batch = torch.stack(data_batch).to(self.args.device)
        target_batch = torch.tensor(target_batch).to(self.args.device)
        true_indices = torch.tensor(true_indices)

        # 计算批次梯度张量
        gradients_tensor = self._batch_compute_gradients(
            theta_t1_prev, data_batch, target_batch
        )

        # 计算批次伪参数张量
        eta_t1 = self.get_lr()
        batch_pseudo_params_tensor = (
            theta_t1_prev.unsqueeze(0) - eta_t1 * gradients_tensor
        )

        # 计算估值
        v_i_t1 = self._batch_compute_valuations(
            theta_t2, theta_t1_prev, batch_pseudo_params_tensor
        )

        # Use true indices for updating cumulative valuations
        for i, true_idx in enumerate(true_indices):
            if self.scores_indices and true_idx.item() not in self.scores_indices:
                continue
            # Placeholder for special handling of known flipped samples
            # For example, one might want to log their valuations separately
            # or apply a different weighting if their influence is being studied.
            # if self.flipped_indices and true_idx.item() in self.flipped_indices:
            #     self.logger.debug(f"Valuating known flipped sample {true_idx.item()} with value {v_i_t1[i]}")
            v_cumulative[true_idx] += v_i_t1[i]

    def _batch_compute_gradients(self, theta_t1_prev_tensor, batch_data, batch_targets):
        """
        使用张量运算计算批次梯度，直接返回张量格式。
        Updated to handle individual sample gradient computation.

        Args:
            theta_t1_prev_tensor (torch.Tensor): 上一时刻的参数张量 [N]
            batch_data (torch.Tensor): 批次输入数据 [B, ...]
            batch_targets (torch.Tensor): 批次目标 [B]

        Returns:
            torch.Tensor: 批次梯度张量 [B, N]，其中 N 是模型总参数数量
        """
        B = batch_data.size(0)
        # 将参数张量转换为参数字典并加载到模型中
        theta_t1_prev = self._tensor_to_dict(
            theta_t1_prev_tensor, self.param_shapes, self.param_sizes
        )
        self.model.load_state_dict(theta_t1_prev)
        self.model.train()

        # 获取模型总参数数量
        N = sum(p.numel() for p in self.model.parameters())

        # Initialize gradient tensor
        all_grads = torch.zeros(B, N, device=self.args.device)

        # Process each sample individually
        for i in range(B):
            # Clear previous gradients
            self.model.zero_grad()

            # Forward pass for single sample
            input_i = batch_data[i : i + 1]
            target_i = batch_targets[i : i + 1]
            output_i = self.model(input_i)
            loss_i = self.criterion(output_i, target_i)

            # Backward pass
            loss_i.backward(retain_graph=True)

            # Collect gradients
            grad_i = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_i.append(param.grad.flatten())
                else:
                    grad_i.append(torch.zeros_like(param).flatten())

            # Store flattened gradients
            all_grads[i] = torch.cat(grad_i)

        return all_grads.detach()

    def _batch_compute_valuations(
        self, theta_t2_tensor, theta_t1_prev_tensor, batch_pseudo_params_tensor
    ):
        """
        使用张量运算计算批次样本的估值。
        Corresponds to Equations 12-14 from the LiveVal paper for v_i^t.
        v_i^t = (||Δθ_t|| - ||u^t_i||) / (||Δθ_t|| + ||u^t_i||)
        where:
            Δθ_t = θ^t_ref - θ_{t-1} (theta_t2_tensor - theta_t1_prev_tensor)
            u^t_i = θ^t_ref - (θ_{t-1} - η_t ∇_θ ℓ(f(x_i; θ_{t-1}), y_i))
                  = θ^t_ref - pseudo_param_i (theta_t2_tensor - batch_pseudo_params_tensor[i])

        Args:
            theta_t2_tensor (torch.Tensor): t2 时刻的参数张量 (θ^t_ref in paper)
            theta_t1_prev_tensor (torch.Tensor): t1-1 时刻的参数张量 (θ_{t-1} in paper)
            batch_pseudo_params_tensor (torch.Tensor): 批次伪参数张量 [B, N] (θ_{t-1} - η_t ∇_θ ℓ for each sample i)

        Returns:
            torch.Tensor: 批次样本的估值 [B]
        """
        # Eq 12: Δθ_t = θ^t_ref - θ_{t-1}
        delta_theta = theta_t2_tensor - theta_t1_prev_tensor  # [N]
        delta_theta_norm = torch.norm(delta_theta)  # 标量

        # Eq 13: u^t_i = θ^t_ref - pseudo_param_i
        # pseudo_param_i = θ_{t-1} - η_t ∇_θ ℓ(f(x_i; θ_{t-1}), y_i)
        # batch_pseudo_params_tensor contains pseudo_param_i for each sample in the batch
        u = theta_t2_tensor.unsqueeze(0) - batch_pseudo_params_tensor  # [B, N]
        u_norm = torch.norm(u, dim=1)  # [B]

        # Eq 14: v^t_i = (||Δθ_t|| - ||u^t_i||)/(||Δθ_t|| + ||u^t_i||)
        # Handle division by zero if (||Δθ_t|| + ||u^t_i||) is zero,
        # or if delta_theta_norm is zero (implies Δθ_t is zero vector).
        denominator = delta_theta_norm + u_norm
        scores = torch.where(
            denominator > 1e-9,  # Avoid division by zero or near-zero
            (delta_theta_norm - u_norm) / denominator,
            torch.zeros_like(u_norm),
        )
        # If delta_theta_norm is also very small (e.g. reference and prev params are same), score should be 0
        scores = torch.where(delta_theta_norm > 1e-9, scores, torch.zeros_like(u_norm))

        return scores

    def _dict_to_tensor(self, param_dict):
        """
        将参数字典转换为单个张量，以及参数形状和大小列表。

        Args:
            param_dict (Dict[str, torch.Tensor]): 模型的状态字典

        Returns:
            (torch.Tensor, List[torch.Size], List[int]): 参数张量，参数形状列表，每个参数的元素数量列表
        """
        param_shapes = []
        param_sizes = []
        tensors = []

        for param in param_dict.values():
            param_shapes.append(param.shape)
            size = param.numel()
            param_sizes.append(size)
            tensors.append(param.view(-1).to(self.args.device))

        # 将所有参数展平后连接成一个张量
        param_tensor = torch.cat(tensors)
        return param_tensor, param_shapes, param_sizes

    def _tensor_to_dict(self, param_tensor, param_shapes, param_sizes):
        """
        将参数张量还原为参数字典。

        Args:
            param_tensor (torch.Tensor): 参数张量
            param_shapes (List[torch.Size]): 参数形状列表
            param_sizes (List[int]): 每个参数的元素数量列表

        Returns:
            Dict[str, torch.Tensor]: 模型的状态字典
        """
        param_dict = {}
        index = 0
        for (name, _), shape, size in zip(
            self.model.state_dict().items(), param_shapes, param_sizes
        ):
            param_flat = param_tensor[index : index + size]
            param = param_flat.view(shape)
            param_dict[name] = param
            index += size
        return param_dict

    def _save_delta_record_to_csv(self, epoch, step, L_t, dot_L, delta, delta_change):
        """
        保存每次delta变化的记录到CSV文件。
        """
        file_path = os.path.join(self.args.save_path, "delta_records.csv")
        file_exists = os.path.exists(file_path)
        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["epoch", "step", "L_t", "dot_L", "delta", "delta_change"]
                )
            writer.writerow([epoch, step, L_t, dot_L, delta, delta_change])

    def _adjust_window_size(self, t, L_prev, L_t, delta, dot_L):
        """
        Calculate loss change rate and adjust window size.

        Args:
            t (int): Current step.
            L_prev (float): Previous loss.
            L_t (float): Current loss.
            delta (int): Current window size.
            dot_L (float): Loss change rate.
        Returns:
            tuple: Updated delta and L_prev.
        """
        delta_change = 0
        if L_prev is not None:
            current_epoch = t // len(self.train_loader)
            current_step = t % len(self.train_loader)
            self.logger.debug(
                f"Loss change rate: {dot_L:.4f} at step {current_step} of epoch {current_epoch}"
            )

            if abs(dot_L) > self.eps_max:
                new_delta = min(delta + self.delta_step, self.delta_max)
                delta_change = new_delta - delta
                delta = new_delta
                self.logger.debug("Increased window size to %d at step %d.", delta, t)
            elif abs(dot_L) < self.eps_min:
                new_delta = max(delta - self.delta_step, self.delta_min)
                delta_change = new_delta - delta
                delta = new_delta
                self.logger.debug("Decreased window size to %d at step %d.", delta, t)
            # 保存delta变化记录（无论是否变化都记录，便于分析）
            self._save_delta_record_to_csv(
                current_epoch, current_step, L_t, dot_L, delta, delta_change
            )

            if self.args.log_level == "DEBUG":
                # save to "savepath/L_{timestamps}.csv"
                with open(
                    f"{self.args.save_path}/L_{self.args.timestamp}.csv", "a"
                ) as f:
                    # if 0, 2 then write arguments
                    if current_epoch == 0 and current_step == 2:
                        arg_line = f"delta_0={self.delta},delta_min={self.delta_min},delta_max={self.delta_max},delta_step={self.delta_step},eps_max={self.eps_max},eps_min={self.eps_min}\n"
                        f.write(arg_line)
                        self.logger.debug(arg_line)
                        f.write("epoch,step,L_t,dot_L,delta\n")
                    line = f"{current_epoch},{current_step},{L_t},{dot_L},{delta}\n"
                    f.write(line)
                    self.logger.debug(line)
        L_prev = L_t
        return delta, L_prev

    def _monitor_resources(self):
        """Monitor GPU and CPU memory usage."""
        self.logger.info("Monitoring resources.")
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # in MB
            torch.cuda.reset_peak_memory_stats()
            self.logger.info(
                "CPU memory usage: %.2f MB, GPU memory usage: %.2f MB",
                cpu_memory,
                gpu_memory,
            )
        else:
            gpu_memory = 0
            self.logger.info(
                "CPU memory usage: %.2f MB, GPU not available.", cpu_memory
            )

        return cpu_memory, gpu_memory

    def _train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, step: int, total_steps: int
    ) -> None:
        """
        Perform a single training step.
        Args:
            inputs: Input batch
            targets: Target batch
            step: Current step
            total_steps: Total number of steps
        """
        if step % 20 == 0:
            self.logger.info(f"Starting training step {step}/{total_steps}.")
        self.model.train()
        self.model_optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.logger.debug("Loss calculated: %.4f", loss.item())
        loss.backward()

        # Store gradients for later use
        self.current_grads = {
            name: param.grad.cpu().clone().detach() if param.grad is not None else None
            for name, param in self.model.named_parameters()
        }

        # Store current parameters
        self.current_epoch_parameters.append(
            {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
        )

        self.model_optimizer.step()
        self.logger.debug("Optimizer step completed.")
        if self.scheduler:
            self.scheduler.step()
            self.logger.debug("Scheduler step completed.")
        self.logger.debug("Training step completed.")

    @override
    def get_scores(self, **kwargs):
        """Get the calculated scores."""
        # We'll implement this to match OTI's implementation
        return self._calculate_scores(True, True, False)

    # Add these missing methods to the AD_OTI class:

    @override
    def _get_current_batch(self):
        """Get current batch of data with indices from IndexedDataset."""
        self.logger.debug("Fetching current batch from IndexedDataset.")

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

    @override
    def _get_train_loader(self):
        """Create and return training data loader with IndexedDataset support."""
        self.logger.info("Creating training data loader.")
        if not isinstance(self.dst_train, IndexedDataset):
            indices = np.arange(len(self.dst_train))
            self.dst_train = IndexedDataset(self.dst_train, indices)
            self.logger.info("Wrapped dataset in IndexedDataset")
        # 优化 DataLoader 配置
        train_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=min(self.args.workers, 4),  # 限制工作进程数
            pin_memory=False,  # 禁用 pin_memory
            collate_fn=custom_collate,
            persistent_workers=False,  # 不保持工作进程
        )
        self.logger.info(
            "Training data loader created with batch size %d and %d workers.",
            self.args.selection_batch,
            min(self.args.workers, 4),
        )
        return train_loader, self.dst_train.indices

    def _save_epoch_scores_to_csv(self, filename, scores_dataframe, message_prefix):
        """Save scores to a CSV file and log the action."""
        scores_path = os.path.join(self.args.save_path, filename)
        scores_dataframe.to_csv(scores_path, index=False)
        self.logger.info(f"{message_prefix}{scores_path}")

    @override
    def _calculate_scores(
        self, use_regularization=False, use_learning_rate=True, use_sliding_window=False
    ):
        """
        实现自适应窗口机制的分数计算，修正批次历史和损失历史。
        """
        self.logger.info("[AD_OTI] Starting score calculation with adaptive window.")
        self.before_run()
        self._initialize_data_loader()
        try:
            init_params_path = os.path.join(self.args.save_path, "initial_params.pt")
            init_params = torch.load(
                init_params_path,
                map_location=lambda storage, loc: (
                    storage.cuda(0) if torch.cuda.is_available() else storage
                ),
                weights_only=False,
            )
            for key, value in init_params.items():
                init_params[key] = value.to(self.args.device)
                del value
            self.model.load_state_dict(init_params)
            self.logger.info(f"Loaded initial model parameters from {init_params_path}")
        except FileNotFoundError:
            self.logger.error(
                f"Initial parameters file not found at {init_params_path}. "
                f"Ensure it's saved before calling _calculate_scores."
            )
        v_cumulative = torch.zeros(self.n_train, device=self.args.device)
        T = len(self.train_loader) * self.epochs
        delta = self.delta
        Q_ref = deque()
        Q_theta = deque()
        L_prev = None
        # 优化：只存批次索引
        batch_indices_history = {}
        loss_history = deque(maxlen=self.delta_max + 1)
        theta_0_tensor, self.param_shapes, self.param_sizes = self._dict_to_tensor(
            init_params
        )
        Q_theta.append((0, theta_0_tensor.to(self.args.device)))
        self.logger.info(
            "Initial model parameters (t=0) saved to Q_theta for score calculation."
        )
        current_step_global = 0
        for epoch in range(self.epochs):
            self.logger.info(f"[AD_OTI Score Calc] Epoch {epoch + 1}/{self.epochs}")
            self._setup_optimizer_scheduler(use_learning_rate)
            self.train_iterator = iter(self.train_loader)
            for batch_idx, (inputs, targets, batch_indices) in enumerate(
                self.train_iterator
            ):
                if current_step_global >= T:
                    break
                current_step_global += 1
                t = current_step_global
                try:
                    pass  # 这里 DataLoader 已经返回 batch，无需异常处理
                except Exception as e:
                    self.logger.error(f"Error getting batch at step {t}: {e}")
                    self.train_iterator = iter(self.train_loader)
                    inputs, targets, batch_indices = next(self.train_iterator)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                self.model.train()
                self.model_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.model_optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                theta_t_tensor, _, _ = self._dict_to_tensor(self.model.state_dict())
                Q_theta.append((t, theta_t_tensor.to(self.args.device)))
                while len(Q_theta) > 0 and Q_theta[0][0] < t - self.delta_max:
                    Q_theta.popleft()
                self.model.eval()
                with torch.no_grad():
                    outputs_eval = self.model(inputs)
                    L_t = self.criterion(outputs_eval, targets).item()
                self.model.train()
                loss_history.append(L_t)
                if len(loss_history) >= delta:
                    L_delta_ago = loss_history[-delta]
                    dot_L = (L_t - L_delta_ago) / delta
                else:
                    dot_L = 0.0
                delta, L_prev = self._adjust_window_size(t, L_prev, L_t, delta, dot_L)
                t_prime = min(t + delta - 1, T)
                Q_ref.append((t, t_prime))
                # 只存批次索引到CPU，节省内存
                if isinstance(batch_indices, torch.Tensor):
                    batch_indices_history[t] = batch_indices.clone().cpu()
                else:
                    batch_indices_history[t] = torch.tensor(batch_indices).clone().cpu()
                while batch_indices_history and min(batch_indices_history.keys()) < t - self.delta_max:
                    del batch_indices_history[min(batch_indices_history.keys())]
                self._process_reference_pairs(
                    v_cumulative, Q_ref, Q_theta, t, batch_indices_history
                )
                if current_step_global >= T:
                    self.logger.info(
                        "[AD_OTI Score Calc] Reached total steps T. Stopping epoch simulation."
                    )
                    break
        self.logger.info(
            f"[AD_OTI Score Calc] Final mean score: {v_cumulative.mean().item():.4f}"
        )
        if self.flipped_indices:
            num_detected = self.count_flipped_in_lowest_scores(
                self.logger, self.args, self.flipped_indices, v_cumulative
            )
            self.logger.info(
                f"[AD_OTI Score Calc] Detected {num_detected}/{len(self.flipped_indices)} flipped samples in lowest scores."
            )
        df_scores = pd.DataFrame(
            {"index": np.arange(self.n_train), "score": v_cumulative.cpu().numpy()}
        )
        if self.flipped_indices:
            df_scores["is_flipped"] = df_scores["index"].isin(self.flipped_indices)
        df_scores = df_scores.sort_values("score", ascending=False)
        self._save_epoch_scores_to_csv(
            "ad_oti_calculated_scores.csv",
            df_scores,
            "[AD_OTI Score Calc] Saved calculated scores to ",
        )
        return v_cumulative.cpu()

    @override
    def count_flipped_in_lowest_scores(self, logger, args, flipped_indices, scores):
        """Count how many flipped samples are in the lowest scores."""
        # Convert scores to CPU for numpy operations
        cpu_scores = scores.cpu().numpy()

        # Get indices of the lowest scores
        lowest_indices = np.argsort(cpu_scores)[: args.num_flip]

        # Count flipped samples in the lowest scores
        num_flipped_in_lowest_scores = sum(
            idx in flipped_indices for idx in lowest_indices
        )

        logger.info(
            f"Number of flipped samples in the lowest {args.num_flip} scores: {num_flipped_in_lowest_scores}"
        )

        # save to result.txt
        with open(os.path.join(args.save_path, "result.txt"), "w") as f:
            ratio = float(num_flipped_in_lowest_scores) / float(args.num_flip)
            f.write(f"{ratio}")

        return num_flipped_in_lowest_scores


# Add AD_OTI to SELECTION_METHODS
SELECTION_METHODS["AD_OTI"] = AD_OTI
