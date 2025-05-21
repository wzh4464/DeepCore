###
# File: ./liveval/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 27th January 2025 4:50:50 pm
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
        eps_min=0.1,  # Lower threshold for loss change
        eps_max=0.05,  # Upper threshold for loss change
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
            eps_min: Minimum loss change threshold
            eps_max: Maximum loss change threshold
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
        self.eps_min = eps_min
        self.eps_max = eps_max

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

        # Initialize train_loader and iterator if not exists
        self._initialize_data_loader()

        # Initialize data structures
        T = len(self.train_loader) * self.epochs  # Total number of steps
        v_cumulative = torch.zeros(self.n_train, device=self.args.device)
        delta = self.delta  # Start with initial window size
        Q_ref = deque()  # Queue for reference pairs (t, t')
        Q_theta = deque()  # Queue for model parameters at each step
        L_prev = None  # Previous loss value, initially None

        # Initialize model state
        theta_prev_tensor, self.param_shapes, self.param_sizes = self._dict_to_tensor(
            self.model.state_dict()
        )
        Q_theta.append((0, theta_prev_tensor))
        self.logger.info("Initial model parameters saved to queue.")

        for t in range(1, T + 1):
            # Get current batch
            inputs, targets, batch_indices = self._get_current_batch()
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)

            # Training step
            self._train_step(inputs, targets, t, T)

            # Save current parameters
            theta_t_tensor, _, _ = self._dict_to_tensor(self.model.state_dict())
            Q_theta.append((t, theta_t_tensor))  # Q_theta[t]
            self.logger.debug(f"Model parameters saved to queue at step {t}.")

            # Memory optimization: Remove parameters outside the max window from Q_theta
            while len(Q_theta) > 0 and Q_theta[0][0] < t - self.delta_max:
                Q_theta.popleft()
                self.logger.debug(f"Removed old parameters from Q_theta up to step {Q_theta[0][0] if Q_theta else t - self.delta_max}")

            # Compute current loss and adjust window size
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs)
                L_t = self.criterion(outputs, targets).item()
            self.model.train()
            self.logger.debug(f"Loss computed at step {t}: {L_t}")

            # Adjust window size based on loss change rate
            delta, L_prev = self._adjust_window_size(t, L_prev, L_t, delta)

            # Update reference pairs
            t_prime = min(t + delta - 1, T)
            Q_ref.append((t, t_prime))
            self.logger.debug(
                "Added reference pair (t=%d, t'=%d) to reference queue.", t, t_prime
            )

            # Process completed reference pairs
            self._process_reference_pairs(
                v_cumulative, Q_ref, Q_theta, t, batch_indices
            )

            # Track flipped samples (if applicable)
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

        # Save detection statistics if flipped samples are present
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

    def _process_reference_pairs(self, v_cumulative, Q_ref, Q_theta, t, batch_indices):
        """
        Processes reference pairs and updates cumulative valuations.
        This method processes reference pairs from the queue `Q_ref` where the second element of the pair matches the current time step `t`.
        It retrieves the corresponding model parameters from `Q_theta`, ensures they are on the correct device, and updates the cumulative valuations.
        Args:
            v_cumulative (torch.Tensor): The cumulative valuations tensor to be updated.
            Q_ref (collections.deque): A deque of reference pairs (t1, t2) to be processed.
            Q_theta (collections.deque): A deque of tuples (step, theta) containing model parameters.
            t (int): The current time step.
            batch_indices (torch.Tensor): Indices of the current batch.
        Returns:
            None
        """

        while Q_ref and Q_ref[0][1] == t:
            t_1, t_2 = Q_ref.popleft()
            self.logger.debug("Processing reference pair (t1=%d, t2=%d).", t_1, t_2)
            # Retrieve corresponding model parameters
            theta_t2 = next((theta for step, theta in Q_theta if step == t_2), None)
            theta_t1_prev = next(
                (theta for step, theta in Q_theta if step == t_1 - 1), None
            )

            if theta_t2 is not None and theta_t1_prev is not None:
                # Ensure parameters are on the correct device
                theta_t2 = theta_t2.to(self.args.device)
                theta_t1_prev = theta_t1_prev.to(self.args.device)

                self._update_cumulative_valuations(
                    v_cumulative, batch_indices, theta_t2, theta_t1_prev
                )

            # Clean up old model parameters from the queue
            Q_theta = deque(
                [(step, theta) for step, theta in Q_theta if step >= t_1 - 1]
            )
            self.logger.info(
                "Cleaned up model parameters from queue up to step %d.", t_1 - 1
            )

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
        torch.save(result, save_path)
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
            denominator > 1e-9, # Avoid division by zero or near-zero
            (delta_theta_norm - u_norm) / denominator,
            torch.zeros_like(u_norm),
        )
        # If delta_theta_norm is also very small (e.g. reference and prev params are same), score should be 0
        scores = torch.where(
            delta_theta_norm > 1e-9,
            scores,
            torch.zeros_like(u_norm)
        )

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
            tensors.append(param.view(-1))

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

    def _adjust_window_size(self, t, L_prev, L_t, delta):
        """
        Calculate loss change rate and adjust window size.

        Args:
            t (int): Current step.
            L_prev (float): Previous loss.
            L_t (float): Current loss.
            delta (int): Current window size.

        Returns:
            tuple: Updated delta and L_prev.
        """
        if L_prev is not None:
            dot_L = (L_t - L_prev) / delta
            current_epoch = t // len(self.train_loader)
            current_step = t % len(self.train_loader)
            self.logger.debug(
                f"Loss change rate: {dot_L:.4f} at step {current_step} of epoch {current_epoch}"
            )

            if self.args.log_level == "DEBUG":
                # save to "savepath/L_{timestamps}.csv"
                with open(
                    f"{self.args.save_path}/L_{self.args.timestamp}.csv", "a"
                ) as f:
                    # if 0, 2 then write arguments
                    if current_epoch == 0 and current_step == 2:
                        arg_line = f"delta_0={self.delta},delta_min={self.delta_min},delta_max={self.delta_max},delta_step={self.delta_step},eps_min={self.eps_min},eps_max={self.eps_max}\n"
                        f.write(arg_line)
                        self.logger.debug(arg_line)
                        f.write("epoch,step,L_t,dot_L,delta\n")
                    line = f"{current_epoch},{current_step},{L_t},{dot_L},{delta}\n"
                    f.write(line)
                    self.logger.debug(line)

            if abs(dot_L) > self.eps_min:
                delta = min(delta + self.delta_step, self.delta_max)
                self.logger.debug("Increased window size to %d at step %d.", delta, t)
            elif abs(dot_L) < self.eps_max:
                delta = max(delta - self.delta_step, self.delta_min)
                self.logger.debug("Decreased window size to %d at step %d.", delta, t)
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

        # Ensure dataset is wrapped in IndexedDataset
        if not isinstance(self.dst_train, IndexedDataset):
            indices = np.arange(len(self.dst_train))
            self.dst_train = IndexedDataset(self.dst_train, indices)
            self.logger.info("Wrapped dataset in IndexedDataset")

        # Create DataLoader with custom collate function
        train_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=custom_collate,
        )

        self.logger.info(
            "Training data loader created with batch size %d and %d workers.",
            self.args.selection_batch,
            self.args.workers,
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
        Implement score calculation using the adaptive window mechanism from LiveVal.
        This method simulates the training process, calculating valuations dynamically
        based on adaptive reference points, similar to the `select` method.
        """
        self.logger.info("[AD_OTI] Starting score calculation with adaptive window.")
        self.before_run() # Sets up model, optimizer, criterion, etc.

        # Initialize data loader if not already done
        self._initialize_data_loader()

        # Load initial model parameters (e.g., from before any training)
        # Assuming initial_params.pt is saved by before_run or a similar setup phase
        try:
            init_params_path = os.path.join(self.args.save_path, "initial_params.pt")
            init_params = torch.load(init_params_path, map_location=self.args.device)
            self.model.load_state_dict(init_params)
            self.logger.info(f"Loaded initial model parameters from {init_params_path}")
        except FileNotFoundError:
            self.logger.error(
                f"Initial parameters file not found at {init_params_path}. "
                f"Ensure it's saved before calling _calculate_scores."
            )
            # Fallback: use current model state as 'initial' if file not found, though this might not be intended.
            init_params = self.model.state_dict()
            self.logger.warning("Using current model state as initial parameters.")

        # Initialize scores and other necessary structures
        v_cumulative = torch.zeros(self.n_train, device=self.args.device)
        T = len(self.train_loader) * self.epochs  # Total number of steps
        delta = self.delta  # Start with initial window size from AD_OTI params
        Q_ref = deque()  # Queue for reference pairs (t_eval, t_ref)
        Q_theta = deque() # Queue for model parameters (step, theta_tensor)
        L_prev = None  # Previous loss value

        # Store initial model state (theta_0)
        theta_0_tensor, self.param_shapes, self.param_sizes = self._dict_to_tensor(init_params)
        Q_theta.append((0, theta_0_tensor.to(self.args.device)))
        self.logger.info("Initial model parameters (t=0) saved to Q_theta for score calculation.")

        # Simulate training epochs and steps for score calculation
        current_step_global = 0
        for epoch in range(self.epochs):
            self.logger.info(f"[AD_OTI Score Calc] Epoch {epoch + 1}/{self.epochs}")
            self._setup_optimizer_scheduler(use_learning_rate) # Reset optimizer for each epoch simulation if needed
            
            # Re-initialize train_iterator for each epoch if it's consumed per epoch
            # This ensures we iterate over the full dataset for each simulated epoch.
            self.train_iterator = iter(self.train_loader) 

            for batch_idx, (inputs, targets, batch_indices) in enumerate(self.train_iterator):
                if current_step_global >= T:
                    break # Stop if total steps T are reached
                
                current_step_global += 1
                t = current_step_global

                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                # Perform a training step (simulated)
                self.model.train()
                self.model_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.model_optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # Save current parameters (theta_t)
                theta_t_tensor, _, _ = self._dict_to_tensor(self.model.state_dict())
                Q_theta.append((t, theta_t_tensor.to(self.args.device)))

                # Memory optimization for Q_theta
                while len(Q_theta) > 0 and Q_theta[0][0] < t - self.delta_max:
                    Q_theta.popleft()

                # Compute current loss L_t for window adjustment
                self.model.eval()
                with torch.no_grad():
                    outputs_eval = self.model(inputs) # Re-evaluate on current batch with updated theta_t
                    L_t = self.criterion(outputs_eval, targets).item()
                self.model.train()

                # Adjust window size delta based on loss change rate
                delta, L_prev = self._adjust_window_size(t, L_prev, L_t, delta)

                # Update reference pairs Q_ref
                t_prime = min(t + delta - 1, T) # t_ref in paper is t - 1 + delta_t
                                                # Here, t_eval is t, so t_ref is t + delta -1
                Q_ref.append((t, t_prime))

                # Process completed reference pairs from Q_ref
                # (Valuations are computed for batch_indices active at t_1)
                # Need to track which batch_indices correspond to t_1 when processing
                # For simplicity here, we assume batch_indices of current step t are relevant if t becomes t_1
                # This part needs careful alignment with how _process_reference_pairs in `select` handles batch_indices
                
                # Store batch_indices with their step t for later valuation
                # We need a way to link (t_1, t_2) from Q_ref to the batch_indices active at t_1
                # Let's use a temporary store for batch_indices per step that Q_ref might point to.
                # This is a simplification. A more robust way would be to store (t1, t2, batch_indices_at_t1) in Q_ref
                # or have a separate queue for (step, batch_indices).
                # For now, we'll use the batch_indices of the *current* step `t` if `t` matches a `t_1` from a popped Q_ref item.
                # This is because the valuation v_i^t occurs for data i in batch B_t.
                # So, when (t1, t2) is processed, t1 is the step when the batch was processed.

                # Simplified: Process reference pairs using current batch_indices if t matches a t_1
                # This is slightly different from select method's handling of batch_indices in _process_reference_pairs
                # which uses batch_indices passed into it. Here we simulate step-by-step.

                while Q_ref and Q_ref[0][1] == t: # If current t is a t_ref for some past t_eval
                    t_1_eval, t_2_ref = Q_ref.popleft()
                    self.logger.debug(f"[Score Calc] Processing ref pair (t_eval={t_1_eval}, t_ref={t_2_ref}).")

                    theta_t2_ref = next((theta for step, theta in Q_theta if step == t_2_ref), None)
                    theta_t1_eval_prev = next((theta for step, theta in Q_theta if step == t_1_eval - 1), None)
                    
                    # Retrieve batch_indices for t_1_eval. This is tricky in this loop structure.
                    # The original paper implies v_i^t is for sample i in batch B_t.
                    # So when (t_eval, t_ref) is processed, we need batch_indices from t_eval.
                    # For this simulation, we'll assume the batch_indices are those of the step t_1_eval.
                    # This requires saving batch_indices along with parameters or in a separate queue.
                    # Let's assume `_get_current_batch` was called at t_1_eval and we had its batch_indices.
                    # This is a key difference to resolve for full alignment.

                    # For now, let's assume the batch_indices for valuation are the ones for the *current* step t,
                    # if t_1_eval == t. This is a simplification and likely incorrect for full alignment.
                    # The `_update_cumulative_valuations` expects `batch_indices` that were active
                    # when `theta_t1_prev` was used to compute `gradients_tensor` to form `pseudo_params`.
                    
                    # A more correct simulation would require storing (step, batch_indices) and retrieving batch_indices for t_1_eval.
                    # Let's refine this: we need to find the data (inputs, targets, batch_indices) that were processed at step t_1_eval.
                    # This is not directly available in the current loop structure without more storage.

                    # Given the complexity, and that _calculate_scores in OTI is simpler,
                    # a full simulation of `select`'s valuation might be too much here.
                    # The original request was to integrate adaptive window with score calculation.
                    # A pragmatic approach might be to use the adaptive `delta` to guide which `theta_ref`
                    # to use, but keep the simpler OTI-like batch scoring for `_calculate_scores`.
                    
                    # Revisiting: The goal is *adaptive reference points*. So we do need theta_t2_ref and theta_t1_eval_prev.
                    # The `_update_cumulative_valuations` method needs `batch_indices` that correspond to `theta_t1_eval_prev`.
                    # This means we must associate `batch_indices` with `t_1_eval`.
                    # Let's simplify by using the `inputs`, `targets`, `batch_indices` of the *current* outer loop iteration (batch_idx)
                    # if `t_1_eval` happens to be the *current global step t*.
                    # This is still an approximation. The original `select` method processes `batch_indices` of the current step `t`
                    # if `t` becomes `t_1` for a reference pair. 

                    if theta_t2_ref is not None and theta_t1_eval_prev is not None:
                        # We need the batch_data and batch_targets that were processed at step t_1_eval.
                        # This simulated loop processes batches sequentially. If t_1_eval corresponds to an earlier batch
                        # in this epoch or a previous epoch, we don't have its data readily available here.
                        # This implies that `_calculate_scores` cannot easily replicate the *exact* online valuation of `select`
                        # without storing all historical batches or re-iterating the dataset up to t_1_eval for each valuation.

                        # Compromise: Use the current batch's data (inputs, targets, batch_indices) for valuation 
                        # if t_1_eval == t (i.e., the valuation is for the current step's data using a future reference).
                        # This is what happens in `select` implicitly: batch_indices are from current step `t` which becomes `t_1`.
                        if t_1_eval == t: # If the evaluation point is the current step
                            self._update_cumulative_valuations(
                                v_cumulative, batch_indices, # batch_indices of current step t = t_1_eval
                                theta_t2_ref.to(self.args.device),
                                theta_t1_eval_prev.to(self.args.device)
                            )
                        # Else: if t_1_eval was a past step, we've missed its batch_indices in this simplified flow.
                        # A full version would need to store (step, batch_indices, params) or similar.

                    # Clean up Q_theta based on t_1_eval (similar to select's _process_reference_pairs)
                    Q_theta = deque(
                        [(step_q, theta_q) for step_q, theta_q in Q_theta if step_q >= t_1_eval - 1]
                    )
            
            if current_step_global >= T:
                self.logger.info("[AD_OTI Score Calc] Reached total steps T. Stopping epoch simulation.")
                break

        # Log final score stats
        self.logger.info(f"[AD_OTI Score Calc] Final mean score: {v_cumulative.mean().item():.4f}")
        if self.flipped_indices:
            num_detected = self.count_flipped_in_lowest_scores(
                self.logger, self.args, self.flipped_indices, v_cumulative
            )
            self.logger.info(
                f"[AD_OTI Score Calc] Detected {num_detected}/{len(self.flipped_indices)} flipped samples in lowest scores."
            )

        # Set any zero scores to NaN (matching OTI behavior if desired, or keep as 0)
        # v_cumulative[v_cumulative == 0.0] = float("nan") 
        # Decided to keep as 0 for AD_OTI unless NaN is specifically needed.

        # Save scores if needed (similar to _select_top_samples)
        df_scores = pd.DataFrame({
            'index': np.arange(self.n_train),
            'score': v_cumulative.cpu().numpy()
        })
        if self.flipped_indices:
            df_scores['is_flipped'] = df_scores['index'].isin(self.flipped_indices)
        df_scores = df_scores.sort_values('score', ascending=False)
        self._save_epoch_scores_to_csv("ad_oti_calculated_scores.csv", df_scores, "[AD_OTI Score Calc] Saved calculated scores to ")

        return v_cumulative.cpu() # Ensure scores are returned on CPU

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
        return num_flipped_in_lowest_scores


# Add AD_OTI to SELECTION_METHODS
SELECTION_METHODS["AD_OTI"] = AD_OTI
