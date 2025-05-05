###
# File: /adult.py
# Created Date: Monday, November 25th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 25th November 2024 3:02:19 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


def Adult(data_path):
    """Initialize the Adult dataset.

    Args:
        data_path (str): Path to the directory containing adult dataset files

    Returns:
        tuple: A tuple containing:
            - channel (int): Number of channels (1 for tabular data)
            - im_size (None): Image size (None for tabular data)
            - num_classes (int): Number of classes (2 for binary classification)
            - class_names (list): List of class names
            - mean (None): Mean value (None for tabular data)
            - std (None): Standard deviation (None for tabular data)
            - dst_train (AdultDataset): Training dataset
            - dst_test (AdultDataset): Test dataset
    """
    channel = 1  # 1-dimensional tabular data
    im_size = None  # No image size for tabular data
    num_classes = 2  # Binary classification
    class_names = [str(c) for c in range(num_classes)]
    mean = None  # No normalization mean
    std = None  # No normalization std

    # Load training data
    train_path = os.path.join(data_path, "adult-training.csv")
    dst_train = AdultDataset(train_path, is_train=True)

    # Load test data
    test_path = os.path.join(data_path, "adult-test.csv")
    dst_test = AdultDataset(test_path, is_train=False)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


class AdultDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        """Initialize the Adult Dataset.

        Args:
            data_path: Path to the data file (either training or test)
            is_train: Boolean indicating if this is training data
        """
        self.data_path = data_path
        self.is_train = is_train
        self._prepare_data()
        self.classes = [str(c) for c in range(2)]  # Binary classification

    def _prepare_data(self):
        # Column names for the adult dataset
        columns = [
            "Age",
            "Workclass",
            "fnlgwt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Native country",
            "Income",
        ]

        # Load data with appropriate parameters
        if self.is_train:
            df = pd.read_csv(self.data_path, names=columns, na_values=["?", " ?"])
        else:
            df = pd.read_csv(
                self.data_path,
                names=columns,
                skiprows=1,  # Skip the first row for test data
                na_values=["?", " ?"],
            )

        # Convert numeric columns to float
        numeric_cols = [
            "Age",
            "fnlgwt",
            "Education-Num",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

        # Apply log1p to fnlgwt
        df["fnlgwt"] = np.log1p(df["fnlgwt"])

        # Handle categorical columns
        categorical_cols = [
            "Workclass",
            "Education",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Native country",
        ]

        # Convert categorical columns to string and fill NaN
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str).str.strip()

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(
            df, columns=categorical_cols, prefix_sep=":", dtype="float32"
        )

        # Convert target (Income) to float32 instead of int64
        df_encoded["Income"] = (
            df_encoded["Income"]
            .map(lambda x: 1.0 if str(x).strip() in {">50K", ">50K."} else 0.0)
            .astype("float32")
        )

        # Extract features and target
        target = df_encoded.pop("Income")

        # Ensure all remaining columns are float32
        for col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("float32")

        # Convert to tensors (both as float32)
        self.data = torch.from_numpy(df_encoded.values.astype(np.float32))
        self.targets = torch.from_numpy(target.values.astype(np.float32))

    def __getitem__(self, index):
        """Return data, target, and index as float tensors."""
        return self.data[index].float(), self.targets[index].float(), index

    def __len__(self):
        return len(self.data)
