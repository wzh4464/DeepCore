###
# File: /oti.py
# Created Date: Friday, August 9th 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 24th August 2024 11:12:25 am
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


class OTI(EarlyTrain):
    """
    Implements the Online Training Influence (OTI) method.

    This class saves model parameters after updating each data point, then uses these parameters
    to calculate a score for each data point. The final selection of the subset is based on these scores.
    """

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        mode="scores",  # [full, stored, scores]
        fractions=[0.8, 0.5, 0.3],  # New parameter for subset fractions
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
        print(f"[OTI] Initial parameters saved to {initial_params_path}")

    def train(self, epoch, list_of_train_idx):
        """
        Get the train index for each epoch.
        Insert a hook to store the data order for each epoch.
        """
        self.epoch_data_orders[epoch] = (
            list_of_train_idx.copy()
        )  # Store the data order for this epoch
        return super().train(epoch, list_of_train_idx)

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        """
        Perform operations after loss calculation, including generating and saving pseudo parameters.
        """
        super().after_loss(outputs, loss, targets, batch_inds, epoch)

        # Generate pseudo update parameters for each data point in the batch (on CPU)
        with torch.no_grad():
            self.pseudo_params_list = []
            for data_idx in batch_inds:
                pseudo_params = {
                    name: (param - self.args.selection_lr * param.grad).cpu()
                    for name, param in self.model.named_parameters()
                    if param.grad is not None
                }
                self.pseudo_params_list.append(
                    {"data_idx": data_idx, "params": pseudo_params}
                )

        # Check if this is the best loss so far
        current_loss = loss.mean().item()
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
            self.save_best_params()

        # Update counters
        self.total_params_processed += len(batch_inds)
        self.current_step += 1

    def save_best_params(self):
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(self.best_params, f)
        print(f"[OTI] Best parameters saved to {best_params_path}")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        """
        Perform actions during the update step, including saving model parameters
        and calculating pseudo parameters for each data point.

        Save:
            - Initial parameters (`initial_params.pt`)
            - Pseudo parameters for each data point (`pseudo_params.pkl`)

        Args:
            outputs: Model outputs.
            loss: Computed loss.
            targets: Ground truth labels.
            epoch: Current epoch number.
            batch_idx: Current batch index.
            batch_size: Size of the current batch.
        """
        # Print progress
        if batch_idx % self.args.print_freq == 0:
            print(
                f"| Epoch [{epoch}/{self.epochs}] Iter[{batch_idx+1}/{(self.n_train // batch_size)+1}]\t\tLoss: {loss.item():.4f}"
            )
            print("[OTI] Saving model parameters...")

        # Save the initial parameters of the batch (on CPU)
        self.current_batch_initial_params = {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
        }
        # Save batch information to disk
        self.save_batch_info(
            epoch, self.current_step, self.current_batch_initial_params, loss.item()
        )

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
            pickle.dump({"loss": loss}, f)

    def after_epoch(self):
        super().after_epoch()

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
                },
                f,
            )

        print(f"[OTI] Parameters and data order saved for epoch {self.current_epoch}")
        self.current_epoch_parameters = []
        self.current_step = 0
        self.current_epoch += 1

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

    def calculate_l2_distance(self, params1, params2, device):
        """Calculate L2 distance between two parameter sets"""
        return sum(
            torch.norm(params1[name].to(device) - params2[name].to(device)).item()
            for name in params1
            if name in params2
        )

    def gpu_worker(self, gpu_id, epochs, return_dict, best_params):
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        local_scores = {}

        for epoch in epochs:
            epoch_dir = os.path.join(self.args.save_path, f"epoch_{epoch}")
            batch_dirs = sorted(
                [d for d in os.listdir(epoch_dir) if d.startswith("batch_")]
            )

            for batch_dir in tqdm(
                batch_dirs, desc=f"GPU {gpu_id} processing epoch {epoch}"
            ):
                batch_path = os.path.join(epoch_dir, batch_dir)

                initial_params = torch.load(
                    os.path.join(batch_path, "initial_params.pt"), map_location=device
                )
                initial_distance = self.calculate_l2_distance(
                    initial_params, best_params, device
                )

                with open(os.path.join(batch_path, "pseudo_params.pkl"), "rb") as f:
                    pseudo_params_list = pickle.load(f)

                for pseudo_param in pseudo_params_list:
                    data_idx = pseudo_param["data_idx"]
                    pseudo_params = {
                        k: v.to(device) for k, v in pseudo_param["params"].items()
                    }
                    pseudo_distance = self.calculate_l2_distance(
                        pseudo_params, best_params, device
                    )

                    score = initial_distance - pseudo_distance

                    if data_idx not in local_scores:
                        local_scores[data_idx] = score
                    else:
                        local_scores[data_idx] += score

                torch.cuda.empty_cache()

        return_dict[gpu_id] = local_scores

    def calculate_scores(self):
        """Calculate scores using single or multiple GPUs"""
        try:
            best_params = self.load_best_params()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("[OTI] Using the current model parameters as the best parameters.")
            best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
        if self.num_gpus <= 1:
            return self.single_gpu_calculate_scores(best_params)
        else:
            return self.multi_gpu_calculate_scores(best_params)

    def single_gpu_calculate_scores(self, best_params):
        """Calculate scores using a single GPU"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        total_scores = {}

        for epoch in range(self.epochs):
            epoch_dir = os.path.join(self.args.save_path, f"epoch_{epoch}")
            batch_dirs = sorted(
                [d for d in os.listdir(epoch_dir) if d.startswith("batch_")]
            )

            for batch_dir in tqdm(batch_dirs, desc=f"Processing epoch {epoch}"):
                batch_path = os.path.join(epoch_dir, batch_dir)

                initial_params = torch.load(
                    os.path.join(batch_path, "initial_params.pt"), map_location=device
                )
                initial_distance = self.calculate_l2_distance(
                    initial_params, best_params, device
                )

                with open(os.path.join(batch_path, "pseudo_params.pkl"), "rb") as f:
                    pseudo_params_list = pickle.load(f)

                for pseudo_param in pseudo_params_list:
                    data_idx = pseudo_param["data_idx"]
                    pseudo_params = {
                        k: v.to(device) for k, v in pseudo_param["params"].items()
                    }
                    pseudo_distance = self.calculate_l2_distance(
                        pseudo_params, best_params, device
                    )

                    score = initial_distance - pseudo_distance

                    if data_idx not in total_scores:
                        total_scores[data_idx] = score
                    else:
                        total_scores[data_idx] += score

                torch.cuda.empty_cache()

        return total_scores

    def multi_gpu_calculate_scores(self, best_params):
        """Calculate scores using multiple GPUs"""
        mp.set_start_method("spawn", force=True)

        manager = mp.Manager()
        return_dict = manager.dict()

        epochs_per_gpu = [
            list(range(i, self.epochs, self.num_gpus)) for i in range(self.num_gpus)
        ]

        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self.gpu_worker,
                args=(gpu_id, epochs_per_gpu[gpu_id], return_dict, best_params),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_scores = {}
        for gpu_scores in return_dict.values():
            for idx, score in gpu_scores.items():
                if idx not in total_scores:
                    total_scores[idx] = score
                else:
                    total_scores[idx] += score

        return total_scores

    def load_scores(self):
        """Load pre-computed scores from file"""
        scores_path = os.path.join(self.args.save_path, "oti_scores.pkl")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Pre-computed scores not found at {scores_path}")
        with open(scores_path, "rb") as f:
            return pickle.load(f)

    def select(self, **kwargs):
        """
        Select the subset based on calculated scores.

        Returns:
            dict: A dictionary containing the selected indices and their scores.
        """

        if self.mode == "full":
            self.run()  # Run the training process
            scores = self.calculate_scores()
        elif self.mode == "stored":
            scores = self.calculate_scores()  # Use stored epoch data
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

        return {"indices": selected_indices, "scores": score_array}

    def finish_run(self):
        """
        Finish the run by saving the best parameters and preserving all intermediate results.

        This method saves the best parameters to a separate file and keeps all the intermediate
        epoch data files for further analysis or debugging purposes.
        """
        if self.best_params is None:
            print("[OTI] Warning: No best parameters were saved during the run.")
            self.best_params = {
                name: param.cpu().clone().detach()
                for name, param in self.model.state_dict().items()
            }
            self.save_best_params()
            print("[OTI] Run finished. All intermediate results have been preserved.")
        else:

            print("[OTI] Best parameters were successfully saved during the run.")

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


# Add OTI to SELECTION_METHODS
SELECTION_METHODS["OTI"] = OTI
