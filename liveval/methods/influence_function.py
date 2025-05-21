###
# File: ./liveval/methods/influence_function.py
# Created Date: Friday, May 9th 2025
# Author: Claude
# -----
# Last Modified: Wednesday, 21st May 2025 9:44:22 am
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, override
import os
from tqdm import tqdm

from .earlytrain import EarlyTrain
from .selection_methods import SELECTION_METHODS
from liveval.datasets.flipped_dataset import IndexedDataset
from torch.utils.data import DataLoader, Subset
from liveval.utils.utils import count_flipped_in_lowest_scores


class InfluenceFunction(EarlyTrain):
    """
    Implements the Influence Function method for data valuation.

    This implementation approximates the effect of removing a training point on
    the model's test loss by simulating parameter perturbation.

    Reference:
    "Understanding Black-box Predictions via Influence Functions"
    Pang Wei Koh and Percy Liang, ICML 2017
    """

    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        num_test_samples=100,
        dst_test=None,
        damping_term=0.01,
        **kwargs,
    ):
        """Initialize the Influence Function method."""
        super().__init__(
            dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs
        )
        self.num_test_samples = num_test_samples
        self.dst_test = dst_test
        self.damping_term = damping_term

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "InfluenceFunction initialized with parameters: fraction=%s, epochs=%s, num_test_samples=%s, damping_term=%s",
            fraction,
            epochs,
            num_test_samples,
            damping_term,
        )

        # 使用基类方法获取特殊索引
        self.flipped_indices = self.get_special_indices("flipped")
        self.scores_indices = self.get_special_indices("selection")

        if self.flipped_indices:
            self.logger.info(f"Tracking {len(self.flipped_indices)} flipped samples")

        if self.scores_indices:
            self.logger.info(f"Computing scores for {len(self.scores_indices)} samples")

    def _initialize_test_loader(self):
        """
        Initialize the test data loader with randomly selected test samples.
        """
        if self.dst_test is None:
            self.logger.warning(
                "No test dataset provided. Using a subset of training dataset for evaluation."
            )
            # If num_test_samples is less than the total, randomly select a subset
            indices = np.random.choice(
                len(self.dst_train),
                min(self.num_test_samples, len(self.dst_train)),
                replace=False,
            )
            self.test_subset = Subset(self.dst_train, indices)
        else:
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
        Initialize data loaders for training and testing.
        """
        super()._initialize_data_loader()
        self._initialize_test_loader()

    @override
    def after_epoch(self):
        """
        After the final epoch, save model checkpoint to use for influence calculation.
        """
        super().after_epoch()

        # Save checkpoint after the model is trained
        if self.current_epoch == self.epochs - 1:
            self.logger.info(f"Saving trained model checkpoint")
            model_path = os.path.join(self.args.save_path, "influence_model.pt")
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Model saved to {model_path}")

    def _compute_gradients(self, inputs, targets):
        """
        Compute gradients of loss with respect to model parameters.

        Args:
            inputs: Input data.
            targets: Target data.

        Returns:
            List of parameter gradients.
        """
        self.model.train()  # Required for gradient computation
        self.model.zero_grad()  # Clear previous gradients

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Collect gradients
        grads = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.clone().detach())
            else:
                grads.append(torch.zeros_like(param))

        return grads

    def _compute_influence_scores(self):
        """
        Compute influence scores using parameter perturbation approach.

        The influence of a training example is estimated by measuring how much
        the test loss changes when we perturb the model in the direction opposite
        to that example's gradient.

        Returns:
            torch.Tensor: Tensor of influence scores.
        """
        self.logger.info("Computing influence scores")

        # Load model parameters
        model_path = os.path.join(self.args.save_path, "influence_model.pt")
        if not os.path.exists(model_path):
            self.logger.warning("Model checkpoint not found, training model first")
            self.before_run()
            self.run()
        else:
            self.before_run()  # Initialize model
            self.model.load_state_dict(torch.load(model_path))

        self.model.to(self.args.device)

        # Initialize data loaders if not already initialized
        self._initialize_data_loader()

        # Compute test loss with current model
        self.model.eval()
        baseline_loss = 0.0
        num_test_samples = 0

        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(
                    self.args.device
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                baseline_loss += loss.sum().item()
                num_test_samples += inputs.size(0)

        baseline_loss /= num_test_samples
        self.logger.info(f"Baseline test loss: {baseline_loss:.6f}")

        # Compute influence for each training example
        influences = torch.zeros(len(self.dst_train))

        # Create generator for processing train data
        train_iter = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Computing influences",
        )

        for batch_idx, (inputs, targets, indices) in train_iter:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            for i in range(inputs.size(0)):
                sample_idx = indices[i].item()

                # Skip if not in scores_indices when specified
                if self.scores_indices and sample_idx not in self.scores_indices:
                    continue

                # Compute gradient for single example
                single_input = inputs[i : i + 1]
                single_target = targets[i : i + 1]

                # Get original parameters
                original_params = [p.clone().detach() for p in self.model.parameters()]

                # Compute gradients
                train_grads = self._compute_gradients(single_input, single_target)

                # Perturb model parameters in the opposite direction of the gradient
                # (simulating removal of this sample)
                with torch.no_grad():
                    for param, grad in zip(self.model.parameters(), train_grads):
                        if param.requires_grad:
                            param.add_(self.damping_term * grad)

                # Compute test loss with perturbed model
                self.model.eval()
                perturbed_loss = 0.0

                with torch.no_grad():
                    for test_inputs, test_targets, _ in self.test_loader:
                        test_inputs = test_inputs.to(self.args.device)
                        test_targets = test_targets.to(self.args.device)
                        test_outputs = self.model(test_inputs)
                        test_loss = self.criterion(test_outputs, test_targets)
                        perturbed_loss += test_loss.sum().item()

                perturbed_loss /= num_test_samples

                # Compute influence as difference in test loss
                influence = perturbed_loss - baseline_loss
                influences[sample_idx] = influence

                # Restore original parameters
                with torch.no_grad():
                    for param, orig_param in zip(
                        self.model.parameters(), original_params
                    ):
                        param.copy_(orig_param)

            # Update progress bar
            if batch_idx % 5 == 0:
                train_iter.set_description(
                    f"Computing influences (batch {batch_idx}/{len(self.train_loader)}, "
                    f"influence range: [{influences.min().item():.6f}, {influences.max().item():.6f}])"
                )

        return influences

    @override
    def select(self, **kwargs):
        """
        Select a subset of training data based on influence function.

        Returns:
            dict: A dictionary containing selected indices and their influence scores.
        """
        self.logger.info("Starting InfluenceFunction selection process")

        # First, train the model if not already trained
        model_path = os.path.join(self.args.save_path, "influence_model.pt")
        if not os.path.exists(model_path):
            self.logger.info("Model not found. Training model...")
            self.before_run()
            self.run()

        # Compute influence scores
        influence_scores = self._compute_influence_scores()

        # Convert to numpy for easier handling
        influence_scores_np = influence_scores.cpu().numpy()

        # Select top examples based on influence scores
        # Higher influence = larger increase in loss when removed = more important
        k = int(self.fraction * len(self.dst_train))
        selected_indices = np.argsort(influence_scores_np)[::-1][:k]

        # Log flipped sample detection if applicable
        if self.flipped_indices:
            # Count detected flipped samples in lowest scores
            num_detected = count_flipped_in_lowest_scores(
                self.logger, self.args, self.flipped_indices, influence_scores
            )
            self.logger.info(
                f"Detected {num_detected} out of {len(self.flipped_indices)} flipped samples"
            )

        # Save results
        result = {
            "indices": selected_indices,
            "scores": influence_scores_np,
            "fraction": self.fraction,
        }

        # Save to file
        save_path = os.path.join(self.args.save_path, "influence_function_result.pt")
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
        # Ensure model is trained
        model_path = os.path.join(self.args.save_path, "influence_model.pt")
        if not os.path.exists(model_path):
            self.before_run()
            self.run()

        # Compute and return influence scores
        scores = self._compute_influence_scores().cpu()

        # Handle specific subset scoring like OTI
        if self.scores_indices:
            self.logger.info(
                f"Returning scores for {len(self.scores_indices)} samples only"
            )
            # Create a zero tensor, only fill scores for the specified subset
            full_scores = torch.zeros(len(self.dst_train))
            for idx in self.scores_indices:
                full_scores[idx] = scores[idx]
            return full_scores.numpy()

        return scores.numpy()


# Register InfluenceFunction in the selection methods dictionary
SELECTION_METHODS["influence_function"] = InfluenceFunction
