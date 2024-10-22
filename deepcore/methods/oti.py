###
# File: /oti.py
# Created Date: Friday, August 9th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 22nd October 2024 3:58:05 pm
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
        """
        Perform actions before the entire run starts.
        This method is called at the beginning of the `run` method.
        """
        super().before_run()
        # Store initial parameters
        self.initial_params = {
            name: param.cpu().clone().detach()
            for name, param in self.model.state_dict().items()
        }

        # Save initial parameters
        initial_params_path = os.path.join(self.args.save_path, "initial_params.pkl")
        with open(initial_params_path, "wb") as f:
            pickle.dump(self.initial_params, f)
        self.logger.info(f"[OTI] Initial parameters saved to {initial_params_path}")

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
        """
        Save model parameters and gradients after loss computation but before optimization step.
        This is the ideal point to capture the model's state and gradients.
        """
        super().after_loss(outputs, loss, targets, batch_inds, epoch)
        
        # Create batch directory
        batch_dir = os.path.join(
            self.args.save_path, f"epoch_{epoch}", f"batch_{self.current_step}"
        )
        os.makedirs(batch_dir, exist_ok=True)

        # Save current model parameters, gradients, and batch information
        step_data = {
            "parameters": {
                name: param.cpu().clone().detach()
                for name, param in self.model.named_parameters()
            },
            "gradients": {
                name: param.grad.cpu().clone().detach() if param.grad is not None else None
                for name, param in self.model.named_parameters()
            },
            "batch_indices": batch_inds,
            "loss": loss.item(),
            "learning_rate": self.get_lr()  # 添加当前学习率信息
        }
        
        # Save step data
        with open(os.path.join(batch_dir, "step_data.pkl"), "wb") as f:
            pickle.dump(step_data, f)

        # Update best parameters if needed
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
                f"| Epoch [{epoch}/{self.epochs}] Step [{self.current_step}] Loss: {loss.item():.4f}"
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
                f"| Epoch [{epoch}/{self.epochs}] Iter[{batch_idx+1}/{(self.n_train // batch_size)+1}]\t\tLoss: {loss.item():.4f}"
            )

    def save_best_params(self):
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(self.best_params, f)
        self.logger.info(f"[OTI] Best parameters saved to {best_params_path}")

    def save_batch_info(self, epoch, batch_idx, initial_params, loss):
        batch_dir = os.path.join(
            self.args.save_path, f"epoch_{epoch}", f"batch_{batch_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)

        # Save initial parameters
        torch.save(initial_params, os.path.join(batch_dir, "initial_params.pt"))

        # Save the pseudo parameter list
        with open(os.path.join(batch_dir, "pseudo_params.pkl"), "wb") as f:
            pickle.dump(self.pseudo_params_list, f)

        # Save other information
        with open(os.path.join(batch_dir, "info.pkl"), "wb") as f:
            pickle.dump({"loss": loss, "lr": self.get_lr()}, f)

        # def run(self):
        #     result = super().run()

        #     # Set up the scheduler after the optimizer is created in the parent's run method
        #     if self.args.scheduler == "CosineAnnealingLR":
        #         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             self.model_optimizer, T_max=self.epochs
        #         )
        #     elif self.args.scheduler == "StepLR":
        #         self.scheduler = torch.optim.lr_scheduler.StepLR(
        #             self.model_optimizer, step_size=30, gamma=0.1
        #         )
        #     # Add more scheduler options as needed

        #     return result

        # def get_lr(self):
        #     return self.model_optimizer.param_groups[0]["lr"]

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
        """Calculate scores using single or multiple GPUs"""
        try:
            best_params = self.load_best_params()
        except FileNotFoundError:
            self.logger.info(
                "[OTI] Using the current model parameters as the best parameters."
            )
            best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }

        if self.num_gpus <= 1:
            return self.single_gpu_calculate_scores(
                best_params,
                use_regularization,
                use_learning_rate,
                # use_sliding_window
            )
        else:
            return self.multi_gpu_calculate_scores(
                best_params,
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

    def calculate_scores_on_device(
        self,
        device_id,
        epochs_to_process,
        best_params,
        use_regularization=False,
        use_learning_rate=True,
        return_dict=None,
    ):
        """
        Calculate scores using specified device (GPU or CPU).

        Args:
            device_id: Device ID to use (-1 for CPU, >=0 for GPU)
            epochs_to_process: List of epochs to process on this device
            best_params: Best model parameters to compare against
            use_regularization: Whether to use regularization in distance calculation
            use_learning_rate: Whether to scale scores by learning rate
            return_dict: Optional multiprocessing manager dict for multi-GPU mode

        Returns:
            Dictionary of scores if single GPU/CPU mode, None if multi-GPU mode
        """
        
        logger = logging.getLogger(__name__)
        logger.info(f"[OTI] Processing epochs {epochs_to_process} on device {device_id}")
        
        # Set up device
        if device_id >= 0:  # GPU mode
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:  # CPU mode
            device = torch.device("cpu")

        local_scores = {}

        # Process each epoch
        for epoch in epochs_to_process:
            epoch_dir = os.path.join(self.args.save_path, f"epoch_{epoch}")
            batch_dirs = sorted(
                [d for d in os.listdir(epoch_dir) if d.startswith("batch_")]
            )

            epoch_lr = self.get_epoch_lr(epoch) if use_learning_rate else 1.0

            # Process each batch
            desc = (
                f"GPU {device_id} processing epoch {epoch}"
                if device_id >= 0
                else f"Processing epoch {epoch}"
            )
            for batch_dir in tqdm(batch_dirs, desc=desc):
                batch_path = os.path.join(epoch_dir, batch_dir)

                # Load step data
                with open(os.path.join(batch_path, "step_data.pkl"), "rb") as f:
                    step_data = pickle.load(f)

                # Move data to device
                current_params = {
                    k: v.to(device) for k, v in step_data["parameters"].items()
                }
                current_grads = {
                    k: v.to(device) if v is not None else None
                    for k, v in step_data["gradients"].items()
                }

                # Calculate pseudo parameters
                pseudo_params = self.calculate_pseudo_params(
                    current_params, current_grads, epoch_lr
                )

                # Calculate distances
                initial_distance = self.calculate_l2_distance(
                    current_params, best_params, device
                )
                pseudo_distance = self.calculate_l2_distance(
                    pseudo_params, best_params, device
                )

                # Update scores for each sample in the batch
                for idx in step_data["batch_indices"]:
                    
                    score = (initial_distance - pseudo_distance) / initial_distance if use_regularization else initial_distance - pseudo_distance

                    if idx not in local_scores:
                        local_scores[idx] = score
                    else:
                        local_scores[idx] += score

                torch.cuda.empty_cache()

        # Handle return based on mode
        if return_dict is not None:  # Multi-GPU mode
            return_dict[device_id] = local_scores
        else:  # Single GPU/CPU mode
            return local_scores

    # if the bottleneck is i/o, we can load all data into memory at once
    """
    def calculate_scores_on_device(
        self,
        device_id,
        epochs_to_process,
        best_params,
        use_regularization=False,
        use_learning_rate=True,
        return_dict=None,
    ):
        if device_id >= 0:
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        local_scores = {}

        for epoch in epochs_to_process:
            epoch_dir = os.path.join(self.args.save_path, f"epoch_{epoch}")
            batch_dirs = sorted(
                [d for d in os.listdir(epoch_dir) if d.startswith("batch_")]
            )

            epoch_lr = self.get_epoch_lr(epoch) if use_learning_rate else 1.0

            # 一次性加载该epoch的所有批次数据
            batch_data = {}
            for batch_dir in batch_dirs:
                batch_path = os.path.join(epoch_dir, batch_dir)
                with open(os.path.join(batch_path, "step_data.pkl"), "rb") as f:
                    batch_data[batch_dir] = pickle.load(f)

            # 每个epoch只将数据传输到设备一次
            for batch_dir, step_data in batch_data.items():
                current_params = {
                    k: v.to(device) for k, v in step_data["parameters"].items()
                }
                current_grads = {
                    k: v.to(device) if v is not None else None
                    for k, v in step_data["gradients"].items()
                }

                pseudo_params = self.calculate_pseudo_params(
                    current_params, current_grads, self.args.selection_lr
                )

                initial_distance = self.calculate_l2_distance(
                    current_params, best_params, device, use_regularization
                )
                pseudo_distance = self.calculate_l2_distance(
                    pseudo_params, best_params, device, use_regularization
                )

                for idx in step_data["batch_indices"]:
                    score = initial_distance - pseudo_distance
                    if use_learning_rate:
                        score *= epoch_lr

                    if idx not in local_scores:
                        local_scores[idx] = score
                    else:
                        local_scores[idx] += score

        if return_dict is not None:
            return_dict[device_id] = local_scores
        else:
            return local_scores
        """

    def single_gpu_calculate_scores(
        self, best_params, use_regularization=False, use_learning_rate=True
    ):
        """Calculate scores using single GPU or CPU"""
        device_id = 0 if torch.cuda.is_available() else -1
        return self.calculate_scores_on_device(
            device_id,
            range(self.epochs),
            best_params,
            use_regularization,
            use_learning_rate,
        )

    def multi_gpu_calculate_scores(
        self, best_params, use_regularization=False, use_learning_rate=True
    ):
        """Calculate scores using multiple GPUs"""
        logger = logging.getLogger(__name__)
        logger.info(f"[OTI] Starting multi-GPU score calculation on {self.num_gpus} GPUs")
        
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()

        # Split epochs among GPUs
        epochs_per_gpu = [
            list(range(i, self.epochs, self.num_gpus)) for i in range(self.num_gpus)
        ]

        # Start processes for each GPU
        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self.calculate_scores_on_device,
                args=(
                    gpu_id,
                    epochs_per_gpu[gpu_id],
                    best_params,
                    use_regularization,
                    use_learning_rate,
                    return_dict,
                ),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Combine results from all GPUs
        total_scores = {}
        for gpu_scores in return_dict.values():
            for idx, score in gpu_scores.items():
                if idx not in total_scores:
                    total_scores[idx] = score
                else:
                    total_scores[idx] += score

        return total_scores

    # Learning Rate Methods
    def get_epoch_lr(self, epoch):
        """Retrieve the learning rate for a specific epoch"""
        epoch_file = os.path.join(self.args.save_path, f"epoch_{epoch}_data.pkl")
        if os.path.exists(epoch_file):
            with open(epoch_file, "rb") as f:
                epoch_data = pickle.load(f)
                return epoch_data.get("learning_rate", 1.0)
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
            self.run()  # Run the training process
            scores = self.calculate_scores(
                use_regularization, use_learning_rate, use_sliding_window
            )
        elif self.mode == "stored":
            scores = self.calculate_scores(
                use_regularization, use_learning_rate, use_sliding_window
            )  # Use stored epoch data
        elif self.mode == "scores":
            scores = self.load_scores()  # Load pre-computed scores
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Convert scores to numpy array
        score_array = np.array(list(scores.values()))
        indices = np.array(list(scores.keys()))

        # Select top-k samples based on the scores
        top_k = self.coreset_size
        selected_indices = indices[np.argsort(score_array)[::-1][:top_k]]

        logger = logging.getLogger(__name__)
        logger.info(f"[OTI] Selected {top_k} samples based on scores.")
        logger.info(f"[OTI] Selected scores: {score_array[selected_indices]}")

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
