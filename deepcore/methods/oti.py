###
# File: /oti.py
# Created Date: Friday, August 9th 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 6th November 2024 5:01:50 pm
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
from .selection_methods import SELECTION_METHODS
from .earlytrain import EarlyTrain
import torch
import numpy as np
import pickle
import torch.multiprocessing as mp
from tqdm import tqdm
import logging
from typing import override
import pandas as pd

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import setup_logging


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
        mode="scores",
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
        self.logger = logging.getLogger(__name__)

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

    @override
    def before_run(self):
        """运行前初始化"""
        super().before_run()
        # 存储初始参数
        self.initial_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }
        # 保存为单个文件
        torch.save(
            self.initial_params,
            os.path.join(self.args.save_path, "initial_params.pt"),  # 使用.pt而不是.pkl
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
            self.scheduler.step()

        file_path = os.path.join(
            self.args.save_path, f"epoch_{self.current_epoch}_data.pkl"
        )

        with open(file_path, "wb") as f:
            pickle.dump(
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
            self.save_best_params()
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
            self.save_best_params()

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

    def save_best_params(self):
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(self.best_params, f)
        # self.logger.info(f"[OTI] Best parameters saved to {best_params_path}")

    def save_batch_info(self, epoch, batch_idx, initial_params, loss):
        batch_dir = os.path.join(
            self.args.save_path, f"epoch_{epoch}", f"batch_{batch_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)

        # Save initial parameters
        torch.save(initial_params, os.path.join(batch_dir, "initial_params.pt"))

    # Parameter Retrieval Methods
    def get_params(self, epoch, step):
        """
        Retrieve parameters for a specific epoch and step.

        Args:
            epoch (int): The epoch number.
            step (int): The step number within the epoch.

        Returns:
            dict: The model parameters at the specified epoch and step.
        """
        epoch_data = self._load_epoch_data(epoch)
        for param_dict in epoch_data["parameters"]:
            if param_dict["step"] == step:
                return param_dict["params"]
        raise ValueError(f"No parameters found for step {step} in epoch {epoch}")

    def get_params_before_after(self, epoch, step):
        """
        Retrieve parameters before and after a specific update step.

        Args:
            epoch (int): The epoch number.
            step (int): The step number within the epoch.

        Returns:
            tuple: A tuple containing two dictionaries (params_before, params_after) and the data point index.
        """
        epoch_data = self._load_epoch_data(epoch)
        params_before = None
        params_after = None
        data_idx = None

        for i, param_dict in enumerate(epoch_data["parameters"]):
            if param_dict["step"] == step:
                params_after = param_dict["params"]
                data_idx = param_dict["data_idx"]
                if i > 0:
                    params_before = epoch_data["parameters"][i - 1]["params"]
                break

        if params_after is None:
            raise ValueError(f"No parameters found for step {step} in epoch {epoch}")

        if params_before is None:
            if epoch <= 0:
                raise ValueError(
                    f"No parameters found before step {step} in epoch {epoch}"
                )

            # If it's the first step of an epoch, get the last step of the previous epoch
            prev_epoch_file = os.path.join(
                self.args.save_path, f"epoch_{epoch-1}_data.pkl"
            )
            with open(prev_epoch_file, "rb") as f:
                prev_epoch_data = pickle.load(f)
            params_before = prev_epoch_data["parameters"][-1]["params"]
        return params_before, params_after, data_idx

    def get_data_order(self, epoch):
        """
        Retrieve the data point order for a specific epoch.

        Args:
            epoch (int): The epoch number.

        Returns:
            list: The order of data points used in the specified epoch.
        """
        epoch_data = self._load_epoch_data(epoch)
        return epoch_data["data_order"]

    def _load_epoch_data(self, epoch):
        """
        Load compressed data for a specific epoch.

        Args:
            epoch (int): The epoch number for which to load the compressed data.

        Returns:
            Any: The data loaded from the compressed file.

        Raises:
            ValueError: If no data is found for the specified epoch.
        """
        file_path = os.path.join(self.args.save_path, f"epoch_{epoch}_data.pkl")
        if not os.path.exists(file_path):
            raise ValueError(f"No data found for epoch {epoch}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def load_best_params(self):
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
            return pickle.load(f)

    # Score Calculation Methods
    def calculate_scores(
        self, use_regularization=False, use_learning_rate=True, use_sliding_window=False
    ):
        """
        Calculate scores by training in real-time and comparing parameters with best_params.
        """
        try:
            best_params = self.load_best_params()
        except FileNotFoundError:
            self.logger.info(
                "[OTI] Using the current model parameters as the best parameters."
            )
            best_params = torch.load(
                os.path.join(self.args.save_path, "best_params.pkl")
            )

        init_params = torch.load(os.path.join(self.args.save_path, "initial_params.pt"))

        if self.num_gpus <= 1:
            self.logger.info("[OTI] Using single GPU for score calculation")
            device_id = self.args.gpu[0]
            return self.single_gpu_calculate_scores(
                best_params,
                init_params,
                device_id,
                use_regularization,
                use_learning_rate,
                # use_sliding_window
            )
        else:
            self.logger.info("[OTI] Using multiple GPUs for score calculation")
            return self.multi_gpu_calculate_scores(
                best_params,
                init_params,
                use_regularization,
                use_learning_rate,
                # use_sliding_window
            )

    def calculate_l2_distance(self, params1, params2, device):
        """Calculate L2 distance between two parameter sets"""
        return sum(
            torch.norm(params1[name].to(device) - params2[name].to(device)).item()
            for name in params1
            if name in params2
        )

    def calculate_pseudo_params(self, params, grads, learning_rate):
        """Calculate pseudo parameters for a given set of parameters and gradients."""
        return {
            name: params[name] - learning_rate * grads[name]
            for name, grad in grads.items()
            if grad is not None
        }

    def _get_train_loader(self):
        """Create and return training data loader."""
        list_of_train_idx = np.random.choice(
            np.arange(self.n_train), self.n_pretrain_size, replace=False
        )

        # 修改 DataLoader，设置 num_workers=0 避免创建子进程
        train_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_sampler=torch.utils.data.BatchSampler(
                list_of_train_idx, batch_size=self.args.selection_batch, drop_last=False
            ),
            num_workers=0,  # 设置为0，不使用多进程加载数据
            pin_memory=True,
        )

        return train_loader, list_of_train_idx

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

    def single_gpu_calculate_scores(
        self,
        best_params,
        init_params,
        device_id,
        use_regularization=False,
        use_learning_rate=True,
        cpus_per_gpu=4,
    ):
        """Calculate scores using single GPU with multiple workers."""
        self.logger.info("[OTI] Starting score calculation on single GPU")
        return_dict = self._init_multiprocessing()

        # Load the model with initial parameters
        self.model.load_state_dict(init_params)
        self.logger.info("[OTI] Loaded initial parameters")

        scores = torch.zeros(self.n_train)
        for name, param in best_params.items():
            best_params[name] = param.to(device_id)

        # Calculate scores
        for epoch in range(self.epochs):
            scores += self.calculate_scores_on_device(
                device_id,
                [epoch],
                best_params,
                use_regularization,
                use_learning_rate,
                return_dict,
            )

        return scores

    def multi_gpu_calculate_scores(
        self,
        best_params,
        use_regularization=False,
        use_learning_rate=True,
        cpus_per_gpu=4,
    ):
        """Calculate scores using multiple GPUs with multiple workers per GPU."""
        raise NotImplementedError("Multi-GPU support is not implemented yet")

    def calculate_scores_on_device(
        self,
        device_id,
        epochs_to_process,
        best_params,
        use_regularization=False,
        use_learning_rate=True,
        return_dict=None,
        worker_id=None,
        train_loader=None,
        train_indices=None,
    ):
        try:
            worker_name = f"Worker-{worker_id}" if worker_id is not None else f"GPU-{device_id}"
            device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

            # 重新创建optimizer和scheduler
            if use_learning_rate:
                # 重用EarlyTrain中的设置
                self.model_optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay
                )
                
                # 重建scheduler
                if self.args.scheduler == "CosineAnnealingLR":
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.model_optimizer, 
                        T_max=self.args.selection_epochs
                    )
                # 可以添加其他scheduler类型
                
            if train_loader is None:
                train_loader, train_indices = self._get_train_loader()

            num_samples = len(train_indices)
            scores = torch.zeros(num_samples, dtype=torch.float32, device="cpu")
            
            for epoch in epochs_to_process:
                # 使用当前epoch的学习率
                if use_learning_rate and self.scheduler:
                    # 设置到正确的epoch
                    for _ in range(epoch):
                        self.scheduler.step()
                    epoch_lr = self.model_optimizer.param_groups[0]['lr']
                else:
                    epoch_lr = 1.0
                    
                self.logger.info(f"[{worker_name}] Epoch {epoch} using lr: {epoch_lr}")

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    local_scores, local_indices = self.process_single_batch(
                        device=device,
                        train_indices=train_indices,
                        best_params=best_params,
                        use_regularization=use_regularization,
                        worker_name=worker_name,
                        epoch_lr=epoch_lr,  # 使用当前epoch的实际学习率
                        epoch=epoch,
                        batch_idx=batch_idx,
                        inputs=inputs,
                        targets=targets,
                    )
                    scores[local_indices] = local_scores.cpu()

            if return_dict is not None:
                return_dict[worker_id if worker_id is not None else device_id] = scores

            return scores

        except Exception as e:
            self.logger.error(f"[{worker_name}] Error: {str(e)}")
            import traceback
            self.logger.error(f"[{worker_name}] Traceback: {traceback.format_exc()}")
            if return_dict is not None:
                return_dict[worker_id if worker_id is not None else device_id] = torch.zeros(num_samples)
            return torch.zeros(num_samples)

    def process_single_batch(
        self,
        inputs,
        targets,
        batch_idx,
        train_indices,
        best_params,
        epoch_lr,
        device,
        use_regularization,
        worker_name,
        epoch,
    ):
        """使用tensor处理batch"""
        # 获取batch索引
        batch_start = batch_idx * self.args.selection_batch
        batch_end = min((batch_idx + 1) * self.args.selection_batch, len(train_indices))
        batch_indices = train_indices[batch_start:batch_end]
        batch_indices_tensor = torch.tensor(batch_indices, device=device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        self.model_optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            # 计算初始距离
            initial_distances = torch.zeros(len(batch_indices), device=device)
            for name, param in self.model.named_parameters():
                if name in best_params:
                    param_diff = param - best_params[name]
                    initial_distances += torch.norm(param_diff.view(1, -1), dim=1)

            # 计算pseudo距离
            pseudo_distances = torch.zeros(len(batch_indices), device=device)
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in best_params:
                    pseudo_param = param - epoch_lr * param.grad
                    param_diff = pseudo_param - best_params[name]
                    pseudo_distances += torch.norm(param_diff.view(1, -1), dim=1)

            # 计算scores
            if use_regularization:
                scores = torch.where(
                    initial_distances > 0,
                    (initial_distances - pseudo_distances) / initial_distances,
                    torch.zeros_like(initial_distances),
                )
            else:
                scores = initial_distances - pseudo_distances

        # 更新参数
        self.model_optimizer.step()

        if batch_idx % 20 == 0:
            self.logger.info(
                f"[{worker_name}] Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}, "
                f"Samples scored: {len(scores)}, "
                f"Mean score: {scores.mean().item():.4f}"
            )

        return scores, batch_indices_tensor

    # Learning Rate Methods
    def get_epoch_lr(self, epoch):
        """Retrieve the learning rate from model's optimizer"""
        try:
            return self.model_optimizer.param_groups[0]["lr"]
        except (AttributeError, IndexError, KeyError):
            self.logger.error("Failed to retrieve learning rate from model's optimizer")
            return 1.0

    def load_scores(self):
        """Load pre-computed scores from file"""
        scores_path = os.path.join(self.args.save_path, "oti_scores.pkl")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Pre-computed scores not found at {scores_path}")
        with open(scores_path, "rb") as f:
            return pickle.load(f)

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
            scores = self.calculate_scores(
                use_regularization, use_learning_rate, use_sliding_window
            )
        elif self.mode == "stored":
            try:
                self.load_stored_params()
                if self.best_params is None:
                    self.logger.error("Failed to load best parameters from stored data")
                    raise ValueError("Failed to load best parameters from stored data")
                scores = self.calculate_scores(
                    use_regularization, use_learning_rate, use_sliding_window
                )
            except FileNotFoundError as e:
                self.logger.error(f"Error loading stored data: {str(e)}")
                raise
        elif self.mode == "scores":
            scores = self.load_scores()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Convert scores to numpy array
        score_array = scores.cpu().numpy()
        indices = torch.arange(self.n_train).cpu().numpy()

        # Create DataFrame with scores
        df = pd.DataFrame({"index": indices, "score": score_array})

        # Sort DataFrame by score in descending order
        df = df.sort_values("score", ascending=False)

        # Save to CSV
        csv_path = os.path.join(self.args.save_path, "oti_scores.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"[OTI] Saved scores to {csv_path}")

        # Select top-k samples based on the scores
        top_k = self.coreset_size
        selected_indices = indices[np.argsort(score_array)[::-1][:top_k]]

        # Save selected indices and their scores
        selected_df = df[df["index"].isin(selected_indices)]
        selected_csv_path = os.path.join(self.args.save_path, "oti_selected_scores.csv")
        selected_df.to_csv(selected_csv_path, index=False)
        self.logger.info(f"[OTI] Saved selected scores to {selected_csv_path}")

        self.logger.info(f"[OTI] Selected {top_k} samples based on scores.")
        self.logger.info(f"[OTI] Selected scores: {score_array[selected_indices]}")

        return {"indices": selected_indices, "scores": score_array}

    # utility methods
    @staticmethod
    def verify_saved_lr(save_path, num_epochs):
        print("[OTI] Starting learning rate verification...")
        for epoch in range(num_epochs):
            epoch_file = os.path.join(save_path, f"epoch_{epoch}_data.pkl")
            if not os.path.exists(epoch_file):
                print(f"[OTI] Warning: File not found for epoch {epoch}")
                continue

            try:
                with open(epoch_file, "rb") as f:
                    epoch_data = pickle.load(f)
            except Exception as e:
                print(f"[OTI] Error reading file for epoch {epoch}: {str(e)}")
                continue

            if "learning_rate" not in epoch_data:
                print(
                    f"[OTI] Warning: Learning rate not found in data for epoch {epoch}"
                )
            else:
                lr = epoch_data["learning_rate"]
                print(f"[OTI] Epoch {epoch} learning rate: {lr}")

        print("[OTI] Learning rate verification completed.")


# Add OTI to SELECTION_METHODS
SELECTION_METHODS["OTI"] = OTI
