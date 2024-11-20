###
# File: ./deepcore/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Sunday, 17th November 2024 12:26:49 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import torch
from .selection_methods import SELECTION_METHODS
from .oti import OTI
from typing import Dict, override
from collections import deque, defaultdict
import numpy as np
import psutil

def dict_add_subtract(dict1, dict2, operation='add', device='cuda'):
    """
    Perform element-wise addition or subtraction on two dictionaries containing tensors.

    Args:
        dict1 (dict): The first dictionary of tensors.
        dict2 (dict): The second dictionary of tensors.
        operation (str): 'add' for addition, 'sub' for subtraction.
        device (str): The device to perform the operations on (default: 'cuda').

    Returns:
        dict: A new dictionary with the resulting tensors.
    """
    assert dict1.keys() == dict2.keys(), "Keys of the dictionaries must match."
    
    # Move tensors to the target device
    dict1 = {k: v.to(device) for k, v in dict1.items()}
    dict2 = {k: v.to(device) for k, v in dict2.items()}
    
    if operation == 'add':
        result = {k: dict1[k] + dict2[k] for k in dict1}
    elif operation == 'sub':
        result = {k: dict1[k] - dict2[k] for k in dict1}
    else:
        raise ValueError("Unsupported operation. Use 'add' or 'sub'.")
    
    return result

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
        # (t, params, learning_rate, batch_indices)
        self.param_queue = deque(
            maxlen=self.delta_max
        )  # FIFO queue for parameters and states
        self.valuation_queue = deque()  # Queue to manage (t, t') pairs
        self.valuations = defaultdict(float)  # Cumulative valuations

        # Initialize state storage
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0

        # Logger
        self.logger = logging.getLogger(__class__.__name__)
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
        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.logger.info("Train loader or iterator not found. Reinitializing.")
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

        # Initialize data structures
        v_cumulative = torch.zeros(len(self.dst_train), device=self.args.device)
        delta = self.delta
        Q_ref = deque()
        Q_theta = deque()
        T = self.epochs * len(self.train_loader)
        L_prev = None

        self.logger.info("Starting main loop.")
        self.logger.info(
            f"Epochs: {self.epochs}, Batches: {len(self.train_loader)}, Total steps: {T}"
        )

        # Initialize model state
        theta_prev = {
            name: param.cpu().clone() for name, param in self.model.state_dict().items()
        }
        Q_theta.append((0, theta_prev))
        self.logger.info("Initial model parameters saved to queue.")

        for t in range(1, T + 1):
            # Get current batch
            inputs, targets, batch_indices = self.get_current_batch()
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)

            # Training step
            self._train_step(inputs, targets, t, T)

            # Save current parameters
            theta_t = {
                name: param.cpu().clone()
                for name, param in self.model.state_dict().items()
            }
            Q_theta.append((t, theta_t))  # Q_theta[t]
            self.logger.debug(f"Model parameters saved to queue at step {t}.")

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
            while Q_ref and Q_ref[0][1] == t:
                t_1, t_2 = Q_ref.popleft()
                self.logger.debug("Processing reference pair (t1=%d, t2=%d).", t_1, t_2)
                # Retrieve corresponding model parameters
                theta_t2 = self._ensure_on_device(
                    next((theta for step, theta in Q_theta if step == t_2), None)
                )
                theta_t1_prev = self._ensure_on_device(
                    next((theta for step, theta in Q_theta if step == t_1 - 1), None)
                )

                if theta_t2 and theta_t1_prev:
                    # Prepare batch data
                    data_batch = torch.stack(
                        [self.dst_train[idx][0] for idx in batch_indices]
                    ).to(self.args.device)
                    target_batch = torch.tensor(
                        [self.dst_train[idx][1] for idx in batch_indices]
                    ).to(self.args.device)

                    # 计算批次梯度
                    batch_gradients = self._batch_compute_gradients(
                        theta_t1_prev, data_batch, target_batch
                    )

                    # 计算批次伪参数
                    eta_t1 = self.get_lr()
                    batch_pseudo_params = self._batch_compute_pseudo_params(
                        theta_t1_prev, batch_gradients, eta_t1
                    )

                    # 计算估值
                    v_i_t1 = self._batch_compute_valuations(
                        theta_t2, theta_t1_prev, batch_pseudo_params, batch_indices
                    )

                    # 更新累积估值
                    v_cumulative[batch_indices] += v_i_t1

                # Clean up old model parameters from the queue
                Q_theta = deque(
                    [(step, theta) for step, theta in Q_theta if step >= t_1 - 1]
                )
                self.logger.info(
                    "Cleaned up model parameters from queue up to step %d.", t_1 - 1
                )

        # After the main loop
        # Select samples based on cumulative valuations
        k = int(self.fraction * len(self.dst_train))
        selected_indices = torch.topk(v_cumulative, k).indices.cpu().numpy()
        self.logger.debug("Selected indices: %s", selected_indices)

        # Save selection results
        result = {
            "indices": selected_indices,
            "valuation": v_cumulative,
            "fraction": self.fraction,
        }

        # Save results to file
        import os

        save_path = os.path.join(self.args.save_path, "selection_result.pt")
        torch.save(result, save_path)
        self.logger.info(f"Saved selection results to {save_path}")
        self.logger.info("Selection process completed.")

        return result

    def _dict_to_tensor(self, param_dict):
        """
        将参数字典转换为单个张量。

        Args:
            param_dict (Dict[str, torch.Tensor]): 参数字典

        Returns:
            tuple: (连接的张量, 每个参数的形状列表, 参数名列表)
        """
        # 使用dict_add_subtract确保所有张量在同一设备上
        param_dict = {k: v.to(self.args.device) for k, v in param_dict.items()}
        param_shapes = []
        param_names = []
        flattened_tensors = []

        for name, param in param_dict.items():
            param_shapes.append(param.shape)
            param_names.append(name)
            # 确保所有张量reshape为 [1, -1]，统一第0维大小为1
            param = param.view(-1).unsqueeze(0)
            flattened_tensors.append(param)

        # 在最后一维连接所有展平的张量
        return torch.cat(flattened_tensors, dim=1), param_shapes, param_names

    def _tensor_to_dict(self, tensor, param_shapes, param_names):
        """
        将单个张量转回参数字典。

        Args:
            tensor (torch.Tensor): 连接的张量
            param_shapes (list): 原始参数形状列表
            param_names (list): 参数名列表

        Returns:
            Dict[str, torch.Tensor]: 参数字典
        """
        param_dict = {}
        start_idx = 0

        for name, shape in zip(param_names, param_shapes):
            flat_size = np.prod(shape[1:]) if len(shape) > 1 else 1
            param_slice = tensor[:, start_idx : start_idx + flat_size]
            param_dict[name] = param_slice.reshape(shape)
            start_idx += flat_size

        return param_dict

    def _batch_compute_valuations(
        self, theta_t2, theta_t1_prev, batch_pseudo_params, batch_indices
    ):
        """
        批量计算样本估值。

        Args:
            theta_t2 (Dict[str, torch.Tensor]): t2时刻的参数
            theta_t1_prev (Dict[str, torch.Tensor]): t1-1时刻的参数
            batch_pseudo_params (Dict[str, torch.Tensor]): 批次伪参数
            batch_indices (tensor): 批次索引

        Returns:
            torch.Tensor: 批次样本的估值
        """
        # 使用dict_add_subtract进行张量计算，加速GPU利用
        delta_theta = dict_add_subtract(theta_t2, theta_t1_prev, operation='sub', device=self.args.device)

        # 将所有字典转换为张量
        theta_t2_tensor, _, _ = self._dict_to_tensor(theta_t2) # torch.Size([1, 61706])
        theta_t1_prev_tensor, _, _ = self._dict_to_tensor(theta_t1_prev) # torch.Size([1, 61706])
        pseudo_params_tensor, _, _ = self._dict_to_tensor(batch_pseudo_params) # torch.Size([1, 15796736])

        # 将伪参数reshape到匹配批次大小 dim 0 = batch_size
        pseudo_params_tensor = pseudo_params_tensor.reshape(len(batch_indices), -1) # torch.Size([256, 61706])

        # 计算delta_theta
        delta_theta = theta_t2_tensor - theta_t1_prev_tensor # torch.Size([1, 61706])

        # 计算范数（在最后一维）
        delta_theta_norm = torch.norm(delta_theta, dim=1) # torch.Size([1])
        delta_theta_norm_expanded = delta_theta_norm.expand(len(batch_indices)) # torch.Size([256])

        # 扩展theta_t2以匹配批次大小 to torch.Size([256, 61706])
        theta_t2_expanded = theta_t2_tensor.expand(len(batch_indices), -1)

        # 计算u
        u = theta_t2_expanded - pseudo_params_tensor # torch.Size([256, 61706])
        u_norm = torch.norm(u, dim=1) # torch.Size([256])

        # 计算估值
        v_i_t1 = (delta_theta_norm_expanded - u_norm) / (delta_theta_norm_expanded + u_norm)

        assert torch.all(v_i_t1 >= -1.0) and torch.all(
            v_i_t1 <= 1.0
        ), "Valuation out of bounds"

        return v_i_t1

    def _ensure_on_device(self, params_dict):
        for name, param in params_dict.items():
            params_dict[name] = param.to(self.args.device)

        return params_dict

    def _expand_parameters(self, params_dict, batch_size):
        """
        将参数字典中的每个参数扩展到指定的批次大小。

        Args:
            params_dict (Dict[str, torch.Tensor]): 参数字典
            batch_size (int): 目标批次大小

        Returns:
            Dict[str, torch.Tensor]: 扩展后的参数字典，每个参数的第一维是batch_size
        """
        expanded_params = {}
        for name, param in params_dict.items():
            # 获取原始形状
            orig_shape = param.shape
            # 创建扩展形状：[batch_size, d1, d2, ...]
            expand_shape = [batch_size] + [-1] * len(orig_shape)

            try:
                # 扩展参数
                expanded = param.unsqueeze(0).expand(expand_shape)
                expanded_params[name] = expanded

                # 调试信息
                self.logger.debug(
                    f"Parameter {name}: "
                    f"original shape {orig_shape}, "
                    f"expanded shape {expanded.shape}"
                )

            except RuntimeError as e:
                self.logger.error(
                    f"Failed to expand parameter {name} "
                    f"from shape {orig_shape} to {expand_shape}"
                )
                raise e

        return expanded_params

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
                # save to "savepath/L_{timestamps}.txt"
                with open(
                    f"{self.args.save_path}/L_{self.args.timestamp}.txt", "a"
                ) as f:
                    f.write(f"{current_epoch},{current_step},{L_t},{dot_L}\n")

            if abs(dot_L) > self.eps_min:
                delta = min(delta + self.delta_step, self.delta_max)
                self.logger.debug("Increased window size to %d at step %d.", delta, t)
            elif abs(dot_L) < self.eps_max:
                delta = max(delta - self.delta_step, self.delta_min)
                self.logger.debug("Decreased window size to %d at step %d.", delta, t)
        L_prev = L_t
        return delta, L_prev

    def _batch_compute_gradients(self, theta_t1_prev, batch_data, batch_targets):
        """
        Compute gradients for a batch of samples simultaneously.

        Args:
            theta_t1_prev: Previous model parameters
            batch_data: Batch of input data [B x ...]
            batch_targets: Batch of targets [B]

        Returns:
            Dict[str, torch.Tensor] with gradients for each parameter
            Each gradient tensor has shape [B x param_shape]
        """
        B = len(batch_data)
        self.model.load_state_dict(theta_t1_prev)

        # Initialize gradient storage
        grad_storage = {
            name: torch.zeros((B,) + param.shape, device=self.args.device)
            for name, param in self.model.named_parameters()
        }

        # Compute gradients for each sample in parallel
        self.model.zero_grad()
        outputs = self.model(batch_data)

        # Compute individual losses for each sample
        individual_losses = torch.stack(
            [
                self.criterion(outputs[i : i + 1], batch_targets[i : i + 1])
                for i in range(B)
            ]
        )

        # Compute gradients for each sample
        for i in range(B):
            individual_losses[i].backward(retain_graph=(i < B - 1))
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_storage[name][i] = param.grad.clone()
            self.model.zero_grad()

        return grad_storage

    def _batch_compute_pseudo_params(self, theta_t1_prev, gradients, eta_t1):
        """
        Compute pseudo parameters for a batch in parallel.

        Args:
            theta_t1_prev: Previous model parameters
            gradients: Batch gradients [B x param_shape]
            eta_t1: Learning rate

        Returns:
            Dict containing pseudo parameters for the batch
        """
        # 使用dict_add_subtract进行伪参数计算，确保在GPU上执行
        adjusted_gradients = {k: eta_t1 * v for k, v in gradients.items()}
        pseudo_params = dict_add_subtract(theta_t1_prev, adjusted_gradients, operation='sub', device=self.args.device)
        return self._ensure_on_device(pseudo_params)

    def _batch_compute_u_i_t(self, theta_t2, pseudo_params):
        """
        Compute u_i_t for a batch in parallel.

        Args:
            theta_t2: Target model parameters
            pseudo_params: Batch of pseudo parameters [B x param_shape]

        Returns:
            Dict containing u_i_t for the batch
        """
        return {
            name: theta_t2[name].unsqueeze(0) - pseudo_params[name] for name in theta_t2
        }

    def _batch_compute_norm(self, param_dict):
        """
        Compute norms for a batch of parameters in parallel.

        Args:
            param_dict: Dict of parameter tensors [B x param_shape]

        Returns:
            torch.Tensor: Batch of norms [B]
        """
        return torch.sqrt(
            sum(
                torch.sum(p**2, dim=tuple(range(1, p.dim())))
                for p in param_dict.values()
            )
        )

    def _compute_norm_dict(
        self, delta_theta: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算给定参数字典中每个参数的范数。

        Args:
            delta_theta (Dict[str, torch.Tensor]): 参数字典。

        Returns:
            Dict[str, float]: 参数范数字典。
        """
        return {name: torch.norm(param).item() for name, param in delta_theta.items()}

    def _calculate_loss_from_theta(self, params, inputs, targets):
        """Calculate loss given a set of parameters."""
        self.logger.debug("Calculating loss from theta.")
        self.model.load_state_dict(params)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.model.train()  # Restore to training mode
        self.logger.debug("Calculated loss: %.4f", loss.item())
        return loss

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

    def _get_train_loader(self):
        """Create and return training data loader."""
        self.logger.info("Creating training data loader.")
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
        return train_loader, list_of_train_idx

    def get_current_batch(self):
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

    def _train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, step: int, total_steps: int
    ) -> None:
        """
        Perform a single training step.
        Args:
            inputs: Input batch
            targets: Target batch
        """
        if step % 20 == 0:
            self.logger.info(f"Starting training step {step}/{total_steps}.")
        self.model.train()
        self.model_optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.logger.debug("Loss calculated: %.4f", loss.item())
        loss.backward()

        self.model_optimizer.step()
        self.logger.debug("Optimizer step completed.")
        if self.scheduler:
            self.scheduler.step()
            self.logger.debug("Scheduler step completed.")
        self.logger.debug("Training step completed.")

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
        # 确保参数在GPU上
        params = {k: v.to(self.args.device) for k, v in params.items()}
        self.logger.debug("Starting gradient computation.")
        self.model.load_state_dict(params)
        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.logger.debug("Backward pass completed.")

        gradients = {
            name: param.grad.cpu().clone().detach()
            for name, param in self.model.named_parameters()
        }
        self.logger.debug("[AD_OTI] Gradient computation completed.")
        return gradients


SELECTION_METHODS["AD_OTI"] = AD_OTI
