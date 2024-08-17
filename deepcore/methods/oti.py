###
# File: /oti.py
# Created Date: Friday, August 9th 2024
# Author: Zihan
# -----
# Last Modified: Friday, 9th August 2024 6:00:28 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
from .selection_methods import SELECTION_METHODS
from .earlytrain import EarlyTrain
import torch
import numpy as np
import pickle

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
        self.args.selection_batch = 1
        self.args.train_batch = 1

        self.current_epoch = 0
        self.current_step = 0
        self.total_params_processed = 0
        self.epoch_data_orders = {}  # To store the data order for each epoch
        self.current_epoch_parameters = []  # To store the parameters for the current epoch

    def train(self, epoch, list_of_train_idx):
        self.epoch_data_orders[epoch] = (
            list_of_train_idx.copy()
        )  # Store the data order for this epoch
        return super().train(epoch, list_of_train_idx)

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        """
        Perform actions during the update step, including saving model parameters
        and calculating incremental scores.

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

        # Store parameters on CPU
        current_params = {name: param.cpu().clone().detach() for name, param in self.model.state_dict().items()}

        self.current_epoch_parameters.append({
            "step": self.current_step,
            "data_idx": self.epoch_data_orders[epoch][batch_idx],
            "params": current_params
        })

        self.total_params_processed += 1
        self.current_step += 1

    def after_epoch(self):
        super().after_epoch()

        # Save parameters and data order for the epoch without compression
        file_path = os.path.join(self.args.save_path, f"epoch_{self.current_epoch}_data.pkl")

        with open(file_path, "wb") as f:
            pickle.dump({
                "parameters": self.current_epoch_parameters,
                "data_order": self.epoch_data_orders[self.current_epoch]
            }, f)

        print(f"[OTI] Parameters and data order saved for epoch {self.current_epoch}")

        # Clear the current epoch parameters from memory
        self.current_epoch_parameters.clear()

        # Reset step counter and increment epoch counter
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
            if epoch > 0:
                # If it's the first step of an epoch, get the last step of the previous epoch
                prev_epoch_file = os.path.join(self.args.save_path, f"epoch_{epoch-1}_data.pkl")
                with open(prev_epoch_file, "rb") as f:
                    prev_epoch_data = pickle.load(f)
                params_before = prev_epoch_data["parameters"][-1]["params"]
            else:
                raise ValueError(f"No parameters found before step {step} in epoch {epoch}")

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

    def calculate_scores(self):
        all_scores = []
        for epoch in range(self.epochs):
            epoch_data = self._load_epoch_data(epoch)
            epoch_parameters = epoch_data["parameters"]

            for i in range(1, len(epoch_parameters)):
                score = sum(
                    torch.norm(epoch_parameters[i]["params"][name] - epoch_parameters[i-1]["params"][name]).item()
                    for name in epoch_parameters[i]["params"]
                    if name in epoch_parameters[i-1]["params"]
                )
                all_scores.append(score)

        return np.array(all_scores)

    def select(self, **kwargs):
        # Run the training process
        self.run()

        # Calculate final scores
        scores = self.calculate_scores()

        # Select top-k samples based on the scores
        selected_indices = np.argsort(scores)[-self.coreset_size :]

        return {"indices": selected_indices, "scores": scores}

    def finish_run(self):
        # Clean up temporary files if needed
        for epoch in range(self.epochs):
            file_path = os.path.join(self.args.save_path, f"epoch_{epoch}_data.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)
        print("[OTI] Cleaned up temporary parameter files")


# Add OTI to SELECTION_METHODS
SELECTION_METHODS["OTI"] = OTI
