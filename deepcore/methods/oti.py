###
 # File: /oti.py
 # Created Date: Friday, August 9th 2024
 # Author: Zihan
 # -----
 # Last Modified: Monday, 19th August 2024 4:42:24 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
 # 2024-08-17        Zihan	Added epoch usage tracking and removed manual memory clearing
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
        self.total_params_processed = 0 # Total number of parameters processed
        self.epoch_data_orders = {} # Store the data order for each epoch
        self.current_epoch_parameters = [] # Store parameters for the current epoch
        self.best_params = None # To store the best parameters
        self.best_loss = float('inf') # To store the best loss
        self.epoch_losses = []  # Track losses for each epoch
        self.epoch_usage = []  # Track whether each epoch was actually used
        self.initial_params = None  # To store initial parameters
        
    def before_run(self):
        """
        Perform actions before the entire run starts.
        This method is called at the beginning of the `run` method.
        """
        super().before_run()
        # Store initial parameters
        self.initial_params = {name: param.cpu().clone().detach() for name, param in self.model.state_dict().items()}
        
        # Save initial parameters
        initial_params_path = os.path.join(self.args.save_path, "initial_params.pkl")
        with open(initial_params_path, "wb") as f:
            pickle.dump(self.initial_params, f)
        print(f"[OTI] Initial parameters saved to {initial_params_path}")
        
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

        # Update best parameters if current loss is lower
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_params = current_params

        self.current_epoch_parameters.append({
            "step": self.current_step,
            "data_idx": self.epoch_data_orders[epoch][batch_idx],
            "params": current_params,
            "loss": loss.item()  # Store the loss separately
        })

        self.total_params_processed += 1
        self.current_step += 1

    def after_epoch(self):
        super().after_epoch()

        # Get the loss of the last parameter update in this epoch
        latest_loss = self.current_epoch_parameters[-1]['loss']
        self.epoch_losses.append(latest_loss)

        # Determine if this epoch was actually used (loss decreased)
        epoch_used = True if len(self.epoch_losses) == 1 else self.epoch_losses[-1] < self.epoch_losses[-2]
        self.epoch_usage.append(epoch_used)

        file_path = os.path.join(self.args.save_path, f"epoch_{self.current_epoch}_data.pkl")

        with open(file_path, "wb") as f:
            pickle.dump({
                "parameters": self.current_epoch_parameters,
                "data_order": self.epoch_data_orders[self.current_epoch],
                "latest_loss": latest_loss,
                "epoch_used": epoch_used
            }, f)

        print(f"[OTI] Parameters and data order saved for epoch {self.current_epoch}")
        print(f"[OTI] Epoch {self.current_epoch} latest loss: {latest_loss:.4f}")
        print(f"[OTI] Epoch {self.current_epoch} used: {epoch_used}")

        # Reset step counter and increment epoch counter
        self.current_step = 0
        self.current_epoch += 1
        self.current_epoch_parameters = []  # Clear the list for the next epoch

        # Note: We no longer manually clear self.current_epoch_parameters

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
                raise ValueError(f"No parameters found before step {step} in epoch {epoch}")

            # If it's the first step of an epoch, get the last step of the previous epoch
            prev_epoch_file = os.path.join(self.args.save_path, f"epoch_{epoch-1}_data.pkl")
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
        """
        Finish the run by saving the best parameters and preserving all intermediate results.

        This method saves the best parameters to a separate file and keeps all the intermediate
        epoch data files for further analysis or debugging purposes.
        """
        # Save the best parameters
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(self.best_params, f)
        print(f"[OTI] Best parameters saved to {best_params_path}")

        # Log the preservation of intermediate results
        print("[OTI] Preserving all intermediate epoch data files for further analysis.")
        
        # List all preserved files and their usage status
        preserved_files = [f for f in os.listdir(self.args.save_path) if f.startswith("epoch_") and f.endswith("_data.pkl")]
        print(f"[OTI] Preserved {len(preserved_files)} epoch data files:")
        for i, file in enumerate(preserved_files):
            print(f"  - {file} (Used: {self.epoch_usage[i]})")

        print("[OTI] Run finished. All intermediate results have been preserved.")

    def load_best_params(self):
        best_params_path = os.path.join(self.args.save_path, "best_params.pkl")
        if os.path.exists(best_params_path):
            with open(best_params_path, "rb") as f:
                return pickle.load(f)
        else:
            print("[OTI] Best parameters file not found.")
            return None

# Add OTI to SELECTION_METHODS
SELECTION_METHODS["OTI"] = OTI
