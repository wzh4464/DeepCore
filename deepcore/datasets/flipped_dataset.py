###
# File: ./deepcore/datasets/flipped_dataset.py
# Created Date: Friday, November 22nd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 25th November 2024 8:26:54 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import torch
import numpy as np


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]
        return data, target, torch.tensor(real_idx, dtype=torch.long)

    def __len__(self):
        return len(self.indices)


class FlippedDataset(IndexedDataset):
    def __init__(self, dst_train, indices, num_scores, num_flip, dataset, seed, logger):
        super().__init__(dst_train, indices)
        self.num_scores = num_scores
        self.num_flip = num_flip
        self.data = dataset
        self.seed = seed
        self.logger = logger
        self.classes = dst_train.classes

        if self.data not in ["mnist", "MNIST", "adult", "Adult", "News20"]:
            raise ValueError(f"Dataset {self.data} is not supported.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Get indices based on dataset type
        if self.data.lower() == "mnist":
            self._flip_mnist_labels()
        elif self.data.lower() == "adult":
            self._flip_adult_labels()
        elif self.data.lower() == "news20":
            self._flip_adult_labels()  # News20 uses same flipping logic as Adult (0 -> 1)

        self.logger.info(
            f"Flipped {self.num_flip} labels from {self.original_label} to {self.target_label}."
        )
        self.logger.info(f"Flipped indices: {self.flipped_indices_permuted}")

    def _flip_mnist_labels(self):
        """Handle MNIST label flipping (1 -> 7)"""
        self._setup_label_flipping(1, 7)

    def _flip_adult_labels(self):
        """Handle Adult dataset label flipping (0 -> 1)"""
        self._setup_label_flipping(0, 1)

    def _setup_label_flipping(self, original_label_value, target_label_value):
        self.original_label = original_label_value
        self.target_label = target_label_value
        self.one_indices = [
            i
            for i, idx in enumerate(self.indices)
            if self.dataset.targets[idx] == self.original_label
        ]
        self._perform_flipping()

    def _perform_flipping(self):
        """Common flipping logic for all datasets"""
        # Select indices to flip
        self.flipped_indices_permuted = np.random.choice(
            self.one_indices, size=self.num_flip, replace=False
        )
        self.flipped_indices_unpermuted = [
            self.indices[idx] for idx in self.flipped_indices_permuted
        ]

        # Initialize scores_indices with flipped indices
        self.scores_indices = self.flipped_indices_unpermuted.copy()

        # Calculate how many additional indices we need
        remaining_indices = list(
            set(self.indices) - set(self.flipped_indices_unpermuted)
        )
        additional_needed = self.num_scores - len(self.flipped_indices_unpermuted)

        if additional_needed > 0:
            # Randomly select additional indices
            additional_indices = np.random.choice(
                remaining_indices,
                size=min(additional_needed, len(remaining_indices)),
                replace=False,
            )
            self.scores_indices.extend(additional_indices)

        # Create flipped targets mapping
        self.flipped_targets = {}
        for idx in self.flipped_indices_permuted:
            self.flipped_targets[self.indices[idx]] = self.target_label

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]

        # Return flipped label if index is in flipped set
        if real_idx in self.flipped_targets:
            target = torch.tensor(self.flipped_targets[real_idx], dtype=target.dtype)
            # 或者确保使用特定类型：
            # target = torch.tensor(self.flipped_targets[real_idx], dtype=torch.float32)

        return data, target, torch.tensor(real_idx, dtype=torch.long)

    def get_flipped_indices(self):
        """
        Retrieve the true indices of the flipped dataset.
        Returns:
            list: A list of indices representing the flipped dataset.
        """
        return self.flipped_indices_unpermuted

    def get_flipped_indices_inner(self):
        """
        Retrieve the permuted indices of the flipped dataset.
        Returns:
            list: A list of indices representing the flipped dataset.
        """
        return self.flipped_indices_permuted

    def get_flipped_selection_from(self):
        """
        Retrieve the true indices of the scores dataset.
        Returns:
            list: A list of indices representing the scores dataset.
        """
        return self.scores_indices
