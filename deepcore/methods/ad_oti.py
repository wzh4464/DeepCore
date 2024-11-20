###
# File: ./deepcore/methods/ad_oti.py
# Created Date: Saturday, November 9th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 19th November 2024 8:03:52 pm
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
from typing import override
from collections import deque, defaultdict
import numpy as np
import psutil


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
        self._initialize_data_loader()

        # Initialize data structures
        v_cumulative, delta, Q_ref, Q_theta, T, L_prev = (
            self.initialize_data_structures()
        )
        # v_cumulative: torch.Tensor, cumulative valuations for each sample
        # delta: int, window size
        # Q_ref: deque, queue for reference pairs (t, t')
        # Q_theta: deque, queue for model parameters at each step
        # T: int, total number of steps
        # L_prev: float, previous loss

        # Initialize model state

        theta_prev_tensor, self.param_shapes, self.param_sizes = self._dict_to_tensor(
            self.model.state_dict()
        )
        Q_theta.append((0, theta_prev_tensor))
        self.logger.info("Initial model parameters saved to queue.")

        for t in range(1, T + 1):
            # Get current batch
            inputs, targets, batch_indices = self.get_current_batch()
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)

            # Training step
            self._train_step(inputs, targets, t, T)

            # Save current parameters
            theta_t_tensor, _, _ = self._dict_to_tensor(self.model.state_dict())
            Q_theta.append((t, theta_t_tensor))  # Q_theta[t]
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
            self.process_reference_pairs(v_cumulative, Q_ref, Q_theta, t, batch_indices)

        return self.select_top_samples(v_cumulative)

    def process_reference_pairs(self, v_cumulative, Q_ref, Q_theta, t, batch_indices):
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

                self.update_cumulative_valuations(
                    v_cumulative, batch_indices, theta_t2, theta_t1_prev
                )

            # Clean up old model parameters from the queue
            Q_theta = deque(
                [(step, theta) for step, theta in Q_theta if step >= t_1 - 1]
            )
            self.logger.info(
                "Cleaned up model parameters from queue up to step %d.", t_1 - 1
            )

    def select_top_samples(self, v_cumulative):
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

    def update_cumulative_valuations(
        self, v_cumulative, batch_indices, theta_t2, theta_t1_prev
    ):
        data_batch = torch.stack([self.dst_train[idx][0] for idx in batch_indices]).to(
            self.args.device
        )
        target_batch = torch.tensor(
            [self.dst_train[idx][1] for idx in batch_indices]
        ).to(self.args.device)

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
        # 更新累积估值
        v_cumulative[batch_indices] += v_i_t1

    def initialize_data_structures(self):
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

        return v_cumulative, delta, Q_ref, Q_theta, T, L_prev

    def _initialize_data_loader(self):
        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.logger.info("Train loader or iterator not found. Reinitializing.")
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

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

    def _batch_compute_gradients(self, theta_t1_prev_tensor, batch_data, batch_targets):
        """
        使用张量运算计算批次梯度，直接返回张量格式。

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

        if hasattr(torch, "vmap"):
            # 使用 vmap 进行向量化计算
            def compute_sample_grad(x, y):
                """计算单个样本的梯度"""
                self.model.zero_grad()
                out = self.model(x.unsqueeze(0))
                loss = self.criterion(out, y.unsqueeze(0))
                # 计算并展平所有参数的梯度
                grads = torch.autograd.grad(
                    loss, self.model.parameters(), create_graph=False
                )
                return torch.cat([g.flatten() for g in grads])

            # 向量化计算函数
            batch_grad_fn = torch.vmap(compute_sample_grad)

            # 并行计算所有样本的梯度
            try:
                all_grads = batch_grad_fn(batch_data, batch_targets)  # [B, N]
            except RuntimeError:  # 如果 vmap 失败，回退到分块处理
                all_grads = torch.zeros(B, N, device=self.args.device)
                chunk_size = 32
                for i in range(0, B, chunk_size):
                    end_idx = min(i + chunk_size, B)
                    all_grads[i:end_idx] = torch.stack(
                        [
                            compute_sample_grad(batch_data[j], batch_targets[j])
                            for j in range(i, end_idx)
                        ]
                    )
        else:
            # 不使用 vmap 时的实现：分块计算
            all_grads = torch.zeros(B, N, device=self.args.device)
            chunk_size = 32  # 可以根据 GPU 内存调整

            for i in range(0, B, chunk_size):
                end_idx = min(i + chunk_size, B)
                chunk_data = batch_data[i:end_idx]
                chunk_targets = batch_targets[i:end_idx]
                chunk_size = end_idx - i

                # 对每个样本计算梯度
                for j in range(chunk_size):
                    self.model.zero_grad()
                    output = self.model(chunk_data[j : j + 1])
                    loss = self.criterion(output, chunk_targets[j : j + 1])

                    # 计算梯度
                    grads = torch.autograd.grad(
                        loss,
                        self.model.parameters(),
                        create_graph=False,
                        retain_graph=False,
                    )

                    # 将所有参数的梯度展平并连接
                    all_grads[i + j] = torch.cat([g.flatten() for g in grads])

        # 确保没有梯度计算图附加到输出
        return all_grads.detach()

    def _batch_compute_valuations(
        self, theta_t2_tensor, theta_t1_prev_tensor, batch_pseudo_params_tensor
    ):
        """
        使用张量运算计算批次样本的估值。

        Args:
            theta_t2_tensor (torch.Tensor): t2 时刻的参数张量
            theta_t1_prev_tensor (torch.Tensor): t1-1 时刻的参数张量
            batch_pseudo_params_tensor (torch.Tensor): 批次伪参数张量 [B, N]

        Returns:
            torch.Tensor: 批次样本的估值 [B]
        """
        delta_theta = theta_t2_tensor - theta_t1_prev_tensor  # [N]
        delta_theta_norm = torch.norm(delta_theta)  # 标量

        u = theta_t2_tensor.unsqueeze(0) - batch_pseudo_params_tensor  # [B, N]
        u_norm = torch.norm(u, dim=1)  # [B]

        return (delta_theta_norm - u_norm) / (delta_theta_norm + u_norm)

    def dict_add_subtract(self, dict1, dict2, operation="add", device="cuda"):
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

        if operation == "add":
            result = {k: dict1[k] + dict2[k] for k in dict1}
        elif operation == "sub":
            result = {k: dict1[k] - dict2[k] for k in dict1}
        else:
            raise ValueError("Unsupported operation. Use 'add' or 'sub'.")

        return result

    def _ensure_on_device(self, params_dict):
        for name, param in params_dict.items():
            params_dict[name] = param.to(self.args.device)

        return params_dict

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


SELECTION_METHODS["AD_OTI"] = AD_OTI
