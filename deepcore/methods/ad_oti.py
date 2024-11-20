###
# File: ./deepcore/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 14th November 2024 9:58:52 pm
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
from typing import Dict, List, override
from collections import deque, defaultdict
import numpy as np
import time
import psutil


class AD_OTI(OTI):
    """
    AD_OTI: Adaptive version of OTI with optimized training and parameter management.
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
            **kwargs,
        )

        # LiveVal hyperparameters
        self.delta = delta_0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_step = delta_step
        self.eps_min = eps_min
        self.eps_max = eps_max

        # Initialize storage for parameters and valuations
        # Each entry in param_queue will be a tuple:
        # (t, params, inputs, targets, batch_indices, learning_rate)
        self.param_queue = deque(
            maxlen=self.delta_max
        )  # FIFO queue for parameters and states
        self.valuations = defaultdict(float)  # Cumulative valuations
        self.time_valuations = defaultdict(list)  # Time series of valuations

        # Initialize state storage
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0

        # Logger
        self.logger = logging.getLogger(__class__.__name__)

    @override
    def select(self, **kwargs):
        """
        Implementation of Algorithm 3: LiveVal with optimized training.

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
        self.logger.debug(f"Total steps: {T}")
        n_samples = len(self.dst_train)
        self.logger.debug(f"Total samples: {n_samples}")
        self.param_queue.clear()  # Clear any previous stored parameters

        # Save initial model parameters as A[0]
        initial_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }
        initial_learning_rate = self.get_lr()
        initial_inputs = None
        initial_targets = None
        initial_batch_indices = None
        self.param_queue.appendleft(
            (
                0,
                initial_params,
                initial_inputs,
                initial_targets,
                initial_batch_indices,
                initial_learning_rate,
            )
        )

        # Initialize valuations
        v_cumulative = torch.zeros(n_samples, device=self.args.device)  # v_i^[0,T]
        v_time_series = [
            torch.zeros(n_samples, device=self.args.device) for _ in range(T)
        ]  # {v_i^[t]}

        # Initialize window size
        delta_t = self.delta

        # Start timing the score calculation
        score_calculation_start = time.time()

        try:
            # Main loop
            for t in range(1, T + 1):
                # Line 5: Get current state (θt-1, ηt, Bt)
                inputs, targets, batch_indices = self.get_current_batch()
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                
                # Train for one step
                current_learning_rate = self.get_lr() # ηt
                self._train_step(inputs, targets) 

                # Store current state parameters and other info in the queue
                current_params = {
                    name: param.cpu().clone().detach()
                    for name, param in self.model.state_dict().items()
                } # θ[t]
                
                self.param_queue.appendleft(
                    (
                        t,
                        current_params,
                        inputs.cpu(),
                        targets.cpu(),
                        batch_indices,
                        current_learning_rate,
                    )
                )

                # If we have enough steps to compute loss change
                if t >= delta_t:
                    ref_t = t - delta_t
                    if ref_entry := next(
                        (entry for entry in self.param_queue if entry[0] == ref_t),
                        None,
                    ):
                        _, ref_params, ref_inputs, ref_targets, ref_batch_indices, _ = (
                            ref_entry
                        )
                        # Compute loss change
                        delta_L = self.compute_loss_change(
                            ref_params, current_params, inputs, targets, delta_t
                        )

                        # Update window size based on delta_L
                        if abs(delta_L) < self.eps_min:
                            delta_t = min(delta_t + self.delta_step, self.delta_max)
                        elif abs(delta_L) > self.eps_max:
                            delta_t = max(delta_t - self.delta_step, self.delta_min)
                    else:
                        # This should not happen as we're maintaining the queue properly
                        self.logger.warning(
                            f"Reference parameters for t={ref_t} not found."
                        )

                # Compute valuations for the current step
                self._compute_batch_valuations(t, v_time_series, v_cumulative)

                # 移除t之前的所有param_queue元素
                self.param_queue = deque(
                    [entry for entry in self.param_queue if entry[0] >= t],
                    maxlen=self.delta_max
                )

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

    def compute_loss_change(
        self,
        ref_params: Dict[str, torch.Tensor],
        current_params: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        delta: int,
    ) -> float:
        """
        Compute loss change between two parameter sets.

        Args:
            ref_params: Parameters at t - delta
            current_params: Parameters at t
            inputs: Input data batch
            targets: Target labels
            delta: Window size

        Returns:
            float: Change in loss divided by delta
        """
        loss_ref = self._calculate_loss_from_theta(ref_params, inputs, targets)
        loss_current = self._calculate_loss_from_theta(current_params, inputs, targets)
        return (loss_current.item() - loss_ref.item()) / delta

    def _calculate_loss_from_theta(self, params, inputs, targets):
        """Calculate loss given a set of parameters."""
        self.model.load_state_dict(params)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.model.train()  # Restore to training mode
        return loss

    def _compute_batch_valuations(
        self,
        t: int,
        v_time_series: List[torch.Tensor],
        v_cumulative: torch.Tensor,
    ) -> None:
        """
        Compute valuations for current batch.
        """
        # Retrieve the current entry from param_queue
        current_entry = next(
            (entry for entry in self.param_queue if entry[0] == t), None
        )
        if not current_entry:
            self.logger.warning(f"Current entry for t={t} not found in param_queue.")
            return

        _, _, inputs, targets, batch_indices, _ = current_entry

        if inputs is None or targets is None or batch_indices is None:
            # Initial step where these are not available
            return

        # Update peak memory usage
        cpu_mem, gpu_mem = self._monitor_resources()
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_mem)
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem)

        # Process each sample in the batch
        for i, idx in enumerate(batch_indices):
            # Get single sample
            input_i = inputs[i : i + 1].to(self.args.device)  # Move to device
            target_i = targets[i : i + 1].to(self.args.device)  # Move to device

            # Retrieve current parameters
            current_params = current_entry[1]

            # Compute gradients
            gradients = self.compute_gradient(current_params, input_i, target_i)

            # Compute pseudo update (u_i^t)
            learning_rate = current_entry[5]
            pseudo_params = {
                name: current_params[name] - learning_rate * gradients[name]
                for name in current_params.keys()
            }

            # Compute u_i^t with the actual update formula
            u_i_t = {
                name: current_params[name] - pseudo_params[name]
                for name in current_params.keys()
            }

            # Compute norms using torch.sqrt(sum of squares) for numerical stability
            def compute_stable_norm(param_dict):
                squared_sum = sum(torch.sum(p * p).item() for p in param_dict.values())
                return torch.sqrt(torch.tensor(squared_sum, device=self.args.device))

            delta_theta_norm = compute_stable_norm(current_params)
            u_i_t_norm = compute_stable_norm(u_i_t)

            # Compute relative improvement and cap it to [0, 1]
            if delta_theta_norm > 0:
                v_i_t = (delta_theta_norm - u_i_t_norm) / delta_theta_norm
                v_i_t = torch.clamp(
                    v_i_t, min=0.0, max=1.0
                )  # Ensure valuation is in [0,1]
            else:
                v_i_t = torch.tensor(0.0, device=self.args.device)

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

    def _monitor_resources(self):
        """Monitor GPU and CPU memory usage."""
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # in MB
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_memory = 0

        return cpu_memory, gpu_memory

    def _get_train_loader(self):
        """Create and return training data loader."""
        # Create a list of indices for training
        list_of_train_idx = np.random.choice(
            np.arange(self.n_train), self.n_pretrain_size, replace=False
        )

        # Create a dataset that returns data along with indices
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

        # Create the indexed dataset
        indexed_dataset = IndexedDataset(self.dst_train, list_of_train_idx)

        # Create DataLoader
        train_loader = torch.utils.data.DataLoader(
            indexed_dataset,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        return train_loader, list_of_train_idx

    def get_current_batch(self):
        """Get current batch of data with indices."""
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

        return inputs, targets, batch_indices

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


SELECTION_METHODS["AD_OTI"] = AD_OTI
