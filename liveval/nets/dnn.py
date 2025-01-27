###
# File: /dnn.py
# Created Date: Monday, November 25th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 25th November 2024 12:33:43 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import logging
import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_input(self, x):
        raise NotImplementedError

    def forward(self, x):
        self.logger.debug(f"BaseModel forward pass with input shape: {x.shape}")
        x = self.preprocess_input(x)
        return self.model(x).squeeze(1)


class DNN(BaseModel):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=128,
        pretrained=False,
        im_size=None,
        logger=None,
    ):
        super(DNN, self).__init__(logger)

        # Calculate total input dimension
        if isinstance(input_dim, (tuple, list, torch.Size)):
            total_input_dim = np.prod(input_dim)
        else:
            total_input_dim = input_dim

        # Determine output dimension based on num_classes
        output_dim = 1 if num_classes == 2 else num_classes

        self.model = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # For binary classification (num_classes=2), no activation needed as we'll use BCEWithLogitsLoss
            # For multi-class, we'll use CrossEntropyLoss which includes LogSoftmax
        )

        self.logger.debug(
            f"Created DNN model with:"
            f"\n - Input dimension: {input_dim}"
            f"\n - Hidden dimension: {hidden_dim}"
            f"\n - Output dimension: {output_dim}"
            f"\n - Classification type: {'binary' if num_classes == 2 else 'multi-class'}"
        )

    def preprocess_input(self, x):
        if len(x.shape) > 2:
            return x.view(x.size(0), -1)
        return x  # Already flat for tabular data

    def param_diff(self, other):
        if not isinstance(other, DNN):
            self.logger.error("Can only compare with another DNN instance")
            raise ValueError("Can only compare with another DNN instance")

        diff = {}
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(), other.named_parameters()
        ):
            if name1 != name2:
                self.logger.error(f"Parameter names do not match: {name1} vs {name2}")
                raise ValueError(f"Parameter names do not match: {name1} vs {name2}")
            diff[name1] = param1.data - param2.data
        return diff

    def param_diff_norm(self, other, norm_type=2):
        diff = self.param_diff(other)
        total_norm = sum(
            torch.norm(diff_tensor, p=norm_type).item() ** norm_type
            for diff_tensor in diff.values()
        )
        return total_norm ** (1 / norm_type)

    def print_param_diff(self, other, threshold=1e-6):
        diff = self.param_diff(other)
        for name, diff_tensor in diff.items():
            if torch.any(torch.abs(diff_tensor) > threshold):
                self.logger.debug(f"Difference in {name}:")
                self.logger.debug(diff_tensor)
