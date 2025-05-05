###
# File: ./liveval/methods/tracin.py
# Created Date: Sunday, May 4th 2025
# Author: Claude
# -----
# Last Modified: Sunday, 4th May 2025 2:00:00 pm
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, override
from collections import defaultdict
import os

from .earlytrain import EarlyTrain
from .selection_methods import SELECTION_METHODS
from liveval.datasets.flipped_dataset import IndexedDataset
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import tqdm


class TracIn(EarlyTrain):
    """
    Implements the TracIn method for influence estimation.

    TracIn computes the influence of a training example by approximating the contribution
    of the example to the gradient of the loss on test examples. It does this by examining
    the gradients at multiple checkpoints during training, tracing how the influence of
    examples evolves throughout the training process.

    Args:
        dst_train: Training dataset.
        args: Configuration arguments.
        fraction: Fraction of data to select (default is 0.5).
        random_seed: Seed for random number generation (default is None).
        epochs: Number of training epochs (default is 200).
        specific_model: Name of a specific model to use (default is None).
        checkpoint_interval: Number of epochs between checkpoints (default is 5).
        num_test_samples: Number of test samples to evaluate influence for (default is 100).
        dst_test: Test dataset to evaluate influence for (default is None).
        aggregation_method: Method for aggregating influence scores ('sum', 'mean', 'max') (default is 'sum').
    """

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        checkpoint_interval=5,
        num_test_samples=100,
        dst_test=None,
        aggregation_method="sum",
        **kwargs,
    ):
        """Initialize the TracIn method."""
        super().__init__(
            dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs
        )
        self.checkpoint_interval = checkpoint_interval
        self.num_test_samples = num_test_samples
        self.aggregation_method = aggregation_method
        self.dst_test = dst_test
        self.checkpoints = []
        self.learning_rates = []
        self.checkpoint_epochs = []

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "TracIn initialized with parameters: fraction=%s, epochs=%s, checkpoint_interval=%s, num_test_samples=%s",
            fraction,
            epochs,
            checkpoint_interval,
            num_test_samples,
        )

        # Track flipped samples if available
        self.flipped_indices = (
            dst_train.get_flipped_indices()
            if hasattr(dst_train, "get_flipped_indices")
            else []
        )

        # 添加对特定子集计算的逻辑，与OTI保持一致
        self.scores_indices = (
            dst_train.get_flipped_selection_from()
            if hasattr(dst_train, "get_flipped_selection_from")
            else []
        )

        if self.flipped_indices:
            self.logger.info(
                f"[TracIn] Tracking {len(self.flipped_indices)} flipped samples"
            )

        # 添加对scores_indices的日志记录
        if self.scores_indices:
            self.logger.info(
                f"[TracIn] Computing scores for {len(self.scores_indices)} samples"
            )

    @override
    def after_epoch(self):
        """
        After each epoch, save model checkpoint if it's the first epoch.
        """
        super().after_epoch()

        # Save checkpoint only at the first epoch
        if self.current_epoch == 0:
            self.logger.info(f"Saving checkpoint at epoch {self.current_epoch}")

            # Save model parameters
            checkpoint = {
                "model_state_dict": {
                    name: param.cpu().clone().detach()
                    for name, param in self.model.state_dict().items()
                },
                "epoch": self.current_epoch,
                "learning_rate": self.get_lr(),
            }

            self.checkpoints.append(checkpoint)
            self.learning_rates.append(self.get_lr())
            self.checkpoint_epochs.append(self.current_epoch)

            # Save checkpoint to file
            checkpoint_path = os.path.join(
                self.args.save_path, f"checkpoint_epoch_{self.current_epoch}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _initialize_test_loader(self):
        """
        Initialize the test data loader with randomly selected test samples.
        """
        if self.dst_test is None:
            self.logger.warning(
                "No test dataset provided. Using training dataset for evaluation."
            )
            self.dst_test = self.dst_train

        # If num_test_samples is less than the total, randomly select a subset
        if self.num_test_samples < len(self.dst_test):
            indices = np.random.choice(
                len(self.dst_test), self.num_test_samples, replace=False
            )
            self.test_subset = Subset(self.dst_test, indices)
        else:
            self.test_subset = self.dst_test

        self.test_loader = DataLoader(
            self.test_subset,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        self.logger.info(
            f"Initialized test loader with {len(self.test_subset)} samples"
        )

    def _initialize_data_loader(self):
        """
        Initializes the data loader for training.
        Make sure:
        - self.train_loader is initialized with the training data.
        - self.train_indices is initialized with the indices of the training data.
        - self.train_iterator is initialized with the iterator for the training loader.
        """
        super()._initialize_data_loader()
        self._initialize_test_loader()

    def _compute_gradients(self, model, inputs, targets, device):
        """
        Compute gradients of loss with respect to model parameters.

        Args:
            model: The model to compute gradients for.
            inputs: Input data.
            targets: Target data.
            device: Device to perform computation on.

        Returns:
            Flattened gradient vector.
        """
        model.train()  # Set model to training mode for gradient computation
        model.zero_grad()  # Clear previous gradients

        # Forward pass
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Collect and flatten gradients
        grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        # Concatenate all gradients into a single vector
        return torch.cat(grads)

    def _compute_influence_single_checkpoint(
        self, checkpoint_idx, train_loader, test_loader, device
    ):
        """
        Compute influence scores for a single checkpoint.

        Args:
            checkpoint_idx: Index of the checkpoint to use.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            device: Device to perform computation on.

        Returns:
            Tensor of influence scores for each training example.
        """
        self.logger.info(f"Computing influence scores for checkpoint {checkpoint_idx}")

        # Load checkpoint model
        checkpoint = self.checkpoints[checkpoint_idx]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        learning_rate = self.learning_rates[checkpoint_idx]

        # Initialize influence scores
        n_train = len(self.dst_train)
        influence_scores = torch.zeros(n_train, device=device)

        # Compute test gradients
        test_gradients = []
        for test_inputs, test_targets, _ in test_loader:
            test_inputs = test_inputs.to(device)
            test_targets = test_targets.to(device)

            for i in range(len(test_inputs)):
                grad = self._compute_gradients(
                    self.model, test_inputs[i : i + 1], test_targets[i : i + 1], device
                )
                test_gradients.append(grad)

        # Average the test gradients
        avg_test_grad = torch.stack(test_gradients).mean(dim=0)

        # Compute influence for each training example
        for inputs, targets, indices in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            for i in range(len(inputs)):
                true_idx_i = indices[i].item()

                # 添加对特定子集计算的逻辑，与OTI保持一致
                # 如果样本不在scores_indices中则跳过
                if self.scores_indices and true_idx_i not in self.scores_indices:
                    continue

                train_grad = self._compute_gradients(
                    self.model, inputs[i : i + 1], targets[i : i + 1], device
                )

                # Influence is the dot product of gradients scaled by learning rate
                influence = learning_rate * torch.dot(train_grad, avg_test_grad)
                influence_scores[indices[i]] = influence.item()

        return influence_scores

    def _compute_influence_scores(self):
        """
        Compute influence scores using only the first checkpoint.

        Returns:
            Tensor of influence scores for each training example.
        """
        device = self.args.device
        self.logger.info(f"Computing influence scores on device: {device}")

        # Initialize data loaders
        self._initialize_data_loader()
        self._initialize_test_loader()

        # Use only the first checkpoint (index 0)
        if len(self.checkpoints) > 0:
            scores = self._compute_influence_single_checkpoint(
                0, self.train_loader, self.test_loader, device
            )
            return scores
        else:
            self.logger.error("No checkpoints available to compute influence scores")
            return torch.zeros(len(self.dst_train), device=device)

    def _parallel_compute_influence(self, num_gpus=1):
        """
        Compute influence scores using only the first checkpoint.
        When using initial epoch only, parallelization is not needed.

        Args:
            num_gpus: Number of GPUs to use for computation (ignored).

        Returns:
            Tensor of influence scores.
        """
        self.logger.info("Using only first checkpoint - falling back to single GPU mode")
        return self._compute_influence_scores()

    @override
    def select(self, **kwargs):
        """
        Select a subset of training data based on TracIn influence scores.

        Returns:
            dict: A dictionary containing selected indices and their influence scores.
        """
        self.logger.info("Starting TracIn selection process")

        # First, train the model to get checkpoints if they don't exist
        if not self.checkpoints:
            self.logger.info("No checkpoints found. Training model...")
            self.before_run()
            self.run()

        # Compute influence scores
        num_gpus = getattr(self.args, "num_gpus", 1)
        if num_gpus > 1:
            self.logger.info(f"Using {num_gpus} GPUs for influence computation")
            influence_scores = self._parallel_compute_influence(num_gpus)
        else:
            influence_scores = self._compute_influence_scores()

        # Convert to numpy for easier handling
        influence_scores = influence_scores.cpu().numpy()

        # Select top examples based on influence scores
        k = int(self.fraction * len(self.dst_train))
        selected_indices = np.argsort(influence_scores)[::-1][:k]

        # Log flipped sample detection if applicable
        if self.flipped_indices:
            detected_flipped = set(selected_indices) & set(self.flipped_indices)
            self.logger.info(
                f"[TracIn] Detected {len(detected_flipped)} out of {len(self.flipped_indices)} flipped samples"
            )

        # self.scores_indices
        self.scores_indices = selected_indices

        # Save results
        result = {
            "indices": selected_indices,
            "scores": influence_scores,
            "fraction": self.fraction,
        }

        # Save to file
        save_path = os.path.join(self.args.save_path, "tracin_result.pt")
        torch.save(result, save_path)
        self.logger.info(f"Saved selection results to {save_path}")

        return result

    @override
    def get_scores(self, **kwargs):
        """
        Get influence scores for all training examples.

        Returns:
            numpy.ndarray: Array of influence scores.
        """
        # Ensure model is trained and checkpoints exist
        if not self.checkpoints:
            self.before_run()
            self.run()

        # Compute and return influence scores
        scores = self._compute_influence_scores().cpu()

        # 添加对特定子集返回的逻辑，与OTI保持一致
        if self.scores_indices:
            self.logger.info(
                f"[TracIn] Returning scores for {len(self.scores_indices)} samples only"
            )
            # 创建一个全零张量，只填充特定子集的得分
            full_scores = torch.zeros(len(self.dst_train))
            for idx in self.scores_indices:
                full_scores[idx] = scores[idx]
            return full_scores.numpy()

        return scores.numpy()


# Register TracIn in the selection methods dictionary
SELECTION_METHODS["TracIn"] = TracIn
