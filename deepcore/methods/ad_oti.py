###
# File: ./deepcore/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 14th November 2024 4:01:20 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import os
import torch
from .selection_methods import SELECTION_METHODS
from .oti import OTI
from typing import Dict, List, Tuple, Optional, override
from collections import defaultdict
import numpy as np
import time


class AD_OTI(OTI):
    """
    AD_OTI: adptive version of OTI
    """

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        delta_0=1,  # Initial window size
        delta_min=1,  # Minimum window size
        delta_max=10,  # Maximum window size
        delta_step=1,  # Window size adjustment step
        eps_min=1e-4,  # Lower threshold for loss change
        eps_max=1e-2,  # Upper threshold for loss change
        **kwargs,
    ):
        """
        Initialize AD_OTI with LiveVal parameters

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
            **kwargs,
        )

        # LiveVal hyperparameters
        self.delta_0 = delta_0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_step = delta_step
        self.eps_min = eps_min
        self.eps_max = eps_max

        # Initialize storage for parameters and valuations
        self.stored_params = {}  # Store model parameters at each step
        self.valuations = defaultdict(float)  # Cumulative valuations
        self.time_valuations = defaultdict(list)  # Time series of valuations

        # Initialize state storage
        self.stored_state = {}
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0

        # logger
        self.logger = logging.getLogger(__class__.__name__)

    def store_params(self, step: int) -> None:
        """Store current model parameters for a given step."""
        self.stored_params[step] = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }

    def compute_loss_change(
        self,
        theta_t_plus_delta: Dict[str, torch.Tensor],
        theta_t_minus_1: Dict[str, torch.Tensor],
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        delta: int,
    ) -> float:
        """
        Compute loss change between two parameter sets.

        Args:
            theta_t_plus_delta: Parameters at t + delta
            theta_t_minus_1: Parameters at t - 1
            batch_inputs: Input data batch
            batch_targets: Target labels
            delta: Window size

        Returns:
            float: Change in loss divided by delta
        """
        loss_1 = self._calculate_loss_from_theta(
            theta_t_plus_delta, batch_inputs, batch_targets
        )
        loss_2 = self._calculate_loss_from_theta(
            theta_t_minus_1, batch_inputs, batch_targets
        )
        return (loss_1.item() - loss_2.item()) / delta

    def _calculate_loss_from_theta(self, arg0, batch_inputs, batch_targets):
        # Load and compute loss with theta_t_plus_delta
        self.model.load_state_dict(arg0)
        outputs_1 = self.model(batch_inputs)
        return self.criterion(outputs_1, batch_targets)

    def compute_gradient(
        self,
        params: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient of loss with respect to parameters.

        Args:
            params: Model parameters
            inputs: Input data
            targets: Target labels

        Returns:
            Dict containing gradients for each parameter
        """
        self.model.load_state_dict(params)
        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        return {
            name: param.grad.cpu().clone().detach()
            for name, param in self.model.named_parameters()
        }

    @override
    def select(self, **kwargs):
        """
        Implementation of Algorithm 3: LiveVal

        Returns:
            dict: Selected indices and their scores
        """
        # Initialize model, optimizer etc.
        self.before_run()

        # Initialize data loaders
        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

        # Calculate total steps and initialize storage
        T = self.epochs * len(self.train_loader)  # Total training steps
        n_samples = len(self.dst_train)
        self.stored_state = {}  # Clear any previous stored states

        # Start timing the score calculation
        score_calculation_start = time.time()

        # Line 1-2: Initialize valuations
        v_cumulative = torch.zeros(n_samples, device=self.args.device)  # v_i^[0,T]
        v_time_series = [
            torch.zeros(n_samples, device=self.args.device) for _ in range(T)
        ]  # {v_i^[t]}

        # Line 3: Initialize δ1
        delta_t = self.delta_0

        try:
            # Line 4: Main loop
            for t in range(1, T + 1):
                # Line 5: Get current state (θt-1, ηt, Bt)
                inputs, targets, batch_indices = self.get_current_batch()
                # Store current state including batch data
                self.store_state(t, inputs, targets, batch_indices)

                # Line 6-19: Adaptive reference point selection
                if t + delta_t <= T:
                    # Train for delta_t steps to get θt-1+δt
                    theta_t_plus_delta = self.train_for_steps(delta_t)

                    # Line 8: Compute ΔL
                    delta_L = (
                        self._calculate_loss_from_theta(
                            theta_t_plus_delta, inputs, targets
                        )
                        - self._calculate_loss_from_theta(
                            self.stored_state[t]["parameters"], inputs, targets
                        )
                    ) / delta_t

                    # Line 9-15: Update window size
                    if abs(delta_L) < self.eps_min:
                        delta_t = min(delta_t + self.delta_step, self.delta_max)
                    elif abs(delta_L) > self.eps_max:
                        delta_t = max(delta_t - self.delta_step, self.delta_min)

                    # Line 16: Set reference parameters
                    theta_ref = theta_t_plus_delta
                elif T in self.stored_state:
                    theta_ref = self.stored_state[T]["parameters"]

                else:
                    # If we don't have the final state yet, use current state
                    theta_ref = self.stored_state[t]["parameters"]
                # Line 20: Compute Δθt
                delta_theta_t = {
                    name: theta_ref[name] - self.stored_state[t]["parameters"][name]
                    for name in theta_ref.keys()
                }

                # Line 21-25: Update valuations for current batch
                self._compute_batch_valuations(
                    t, theta_ref, delta_theta_t, v_time_series, v_cumulative
                )

                # Train for one step
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                self._train_step(inputs, targets)

                # Optional: Log progress
                if t % 100 == 0:
                    self.logger.info(
                        f"Step {t}/{T}, Current window size (δt): {delta_t}"
                    )
                    self.logger.info(
                        f"Current mean valuation: {v_cumulative.mean().item():.4f}"
                    )

        except Exception as e:
            self.logger.error(f"Error during selection: {str(e)}")
            raise

        finally:
            # Calculate score computation time
            score_calculation_time = time.time() - score_calculation_start

            # Save resource usage metrics
            metrics = {
                "score_calculation_time": score_calculation_time,
                "peak_cpu_memory_mb": self.peak_cpu_memory,
                "peak_gpu_memory_mb": self.peak_gpu_memory,
            }

            # Save metrics to file
            metrics_path = os.path.join(self.args.save_path, "resource_metrics.pt")
            torch.save(metrics, metrics_path)

            # Log metrics
            self.logger.info("Resource usage metrics:")
            self.logger.info(
                f"Score calculation time: {score_calculation_time:.2f} seconds"
            )
            self.logger.info(f"Peak CPU memory: {self.peak_cpu_memory:.2f} MB")
            self.logger.info(f"Peak GPU memory: {self.peak_gpu_memory:.2f} MB")

        # Select top-k samples based on cumulative valuations
        k = int(self.fraction * n_samples)
        selected_indices = torch.topk(v_cumulative, k).indices.cpu().numpy()

        # Save selection results
        result = {
            "indices": selected_indices,
            "scores": v_cumulative.cpu().numpy(),
            "time_valuations": [v.cpu().numpy() for v in v_time_series],
        }

        # Save results to file
        save_path = os.path.join(self.args.save_path, "selection_result.pt")
        torch.save(result, save_path)
        self.logger.info(f"Saved selection results to {save_path}")

        return result

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Perform a single training step.
        Args:
            inputs: Input batch
            targets: Target batch
        """
        self.model.train()
        self.model_optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.model_optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def train_for_steps(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        Train the model for a specified number of steps and return parameters.

        Args:
            num_steps: Number of training steps

        Returns:
            dict: Model parameters after training
        """
        original_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }

        for _ in range(num_steps):
            try:
                inputs, targets, _ = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                inputs, targets, _ = next(self.train_iterator)

            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)
            self._train_step(inputs, targets)

        final_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }

        # Restore original parameters
        self.model.load_state_dict(original_params)

        return final_params

    def _get_train_loader(self):
        """Create and return training data loader."""
        # 创建索引列表
        list_of_train_idx = np.random.choice(
            np.arange(self.n_train), self.n_pretrain_size, replace=False
        )

        # 创建一个包装数据集来返回索引
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __getitem__(self, idx):
                true_idx = self.indices[idx]
                data, target = self.dataset[true_idx]
                return data, target, true_idx

            def __len__(self):
                return len(self.indices)

        # 创建带索引的数据集
        indexed_dataset = IndexedDataset(self.dst_train, list_of_train_idx)

        # 创建 DataLoader
        train_loader = torch.utils.data.DataLoader(
            indexed_dataset,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        return train_loader, list_of_train_idx

    def get_current_batch(self):
        """Get current batch of data with indices"""
        # Initialize train_loader and iterator if not exists
        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

        try:
            # Get next batch from iterator
            inputs, targets, batch_indices = next(self.train_iterator)
        except StopIteration:
            # If iterator is exhausted, create new one
            self.train_iterator = iter(self.train_loader)
            inputs, targets, batch_indices = next(self.train_iterator)

        return inputs.to(self.args.device), targets.to(self.args.device), batch_indices

    def _compute_batch_valuations(
        self,
        t: int,
        theta_ref: Dict[str, torch.Tensor],
        delta_theta_t: Dict[str, torch.Tensor],
        v_time_series: List[torch.Tensor],
        v_cumulative: torch.Tensor,
    ) -> None:
        """
        Compute valuations for current batch
        """
        # Get batch data directly from stored state
        batch_indices = self.stored_state[t]["batch_indices"]
        inputs = self.stored_state[t].get("inputs")
        targets = self.stored_state[t].get("targets")

        # Update peak memory usage
        cpu_mem, gpu_mem = self._monitor_resources()
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_mem)
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem)

        # Process each sample in the batch
        for i, idx in enumerate(batch_indices):
            # Get single sample
            input_i = inputs[i : i + 1]  # Already a tensor with correct shape
            target_i = targets[i : i + 1]  # Already a tensor with correct shape

            # Compute gradients
            gradients = self.compute_gradient(
                self.stored_state[t]["parameters"], input_i, target_i
            )

            # Compute pseudo update (u_i^t)
            learning_rate = self.stored_state[t]["learning_rate"]
            pseudo_params = {
                name: self.stored_state[t]["parameters"][name]
                - learning_rate * gradients[name]
                for name in theta_ref
            }

            # Compute u_i^t with the actual update formula
            u_i_t = {name: theta_ref[name] - pseudo_params[name] for name in theta_ref}

            # Compute norms using torch.sqrt(sum of squares) for numerical stability
            def compute_stable_norm(param_dict):
                squared_sum = sum(torch.sum(p * p).item() for p in param_dict.values())
                return torch.sqrt(torch.tensor(squared_sum))

            delta_theta_norm = compute_stable_norm(delta_theta_t)
            u_i_t_norm = compute_stable_norm(u_i_t)

            # Compute relative improvement and cap it to [0, 1]
            if delta_theta_norm > 0:
                v_i_t = (delta_theta_norm - u_i_t_norm) / delta_theta_norm
                v_i_t = torch.clamp(
                    v_i_t, min=0.0, max=1.0
                )  # Ensure valuation is in [0,1]
            else:
                v_i_t = torch.tensor(0.0)

            # Optional: Add debug logging periodically
            if t % 100 == 0 and i == 0:
                self.logger.debug(
                    f"t={t}, idx={idx}, delta_norm={delta_theta_norm:.4f}, "
                    f"u_norm={u_i_t_norm:.4f}, v_i_t={v_i_t:.4f}"
                )

            # Update valuations
            v_time_series[t - 1][idx] = v_i_t
            v_cumulative[idx] += v_i_t

        # Optional: Check for numerical issues
        if torch.any(torch.isnan(v_cumulative)):
            self.logger.warning(f"NaN detected in valuations at step {t}")
        if torch.any(v_cumulative < 0):
            self.logger.warning(f"Negative valuations detected at step {t}")

    def store_state(
        self,
        t: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        batch_indices: List[int] = None,
    ) -> None:
        """Store current model state (θt-1, ηt, Bt) as A[t]"""
        self.stored_state[t] = {
            "parameters": {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            },
            "learning_rate": self.get_lr(),
            "batch_indices": batch_indices,
            "inputs": inputs,
            "targets": targets,
        }

        if t % 20 == 0:
            self.logger.debug(f"Stored state at step {t} to self.stored_state")

    def _monitor_resources(self):
        """监控 GPU 和 CPU 内存使用"""
        import psutil
        import torch

        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_memory = 0

        return cpu_memory, gpu_memory


SELECTION_METHODS["AD_OTI"] = AD_OTI
