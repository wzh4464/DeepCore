###
# File: ./liveval/datasets/news.py
# Created Date: Monday, November 25th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 27th January 2025 4:50:48 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def News20(data_path):
    """Initialize the News20 dataset.

    Args:
        data_path (str): Path to the directory for caching data

    Returns:
        tuple: A tuple containing:
            - channel (int): Number of channels (1 for tabular data)
            - im_size (None): Image size (None for tabular data)
            - num_classes (int): Number of classes (2 for binary classification)
            - class_names (list): List of class names
            - mean (None): Mean value (None for tabular data)
            - std (None): Standard deviation (None for tabular data)
            - dst_train (NewsDataset): Training dataset
            - dst_test (NewsDataset): Test dataset
    """
    channel = 1  # 1-dimensional tabular data
    im_size = None  # No image size for tabular data
    num_classes = 2  # Binary classification
    class_names = [str(c) for c in range(num_classes)]
    mean = None  # No normalization mean
    std = None  # No normalization std

    # 确保数据目录存在
    os.makedirs(data_path, exist_ok=True)

    # 创建训练集和测试集
    dst_train = NewsDataset(data_path, is_train=True)
    dst_test = NewsDataset(data_path, is_train=False)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


class NewsDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        """Initialize the News Dataset.

        Args:
            data_path: Path to store/load the cached data
            is_train: Boolean indicating if this is training data
        """
        self.data_path = data_path
        self.is_train = is_train
        self.cache_file = os.path.join(data_path, "news_data.pkl")
        self._prepare_data()
        self.classes = [str(c) for c in range(2)]  # Binary classification

    def _prepare_data(self):
        if os.path.exists(self.cache_file):
            # 从缓存加载数据
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                if self.is_train:
                    data = cached_data["train_data"]
                    targets = cached_data["train_targets"]
                else:
                    data = cached_data["test_data"]
                    targets = cached_data["test_targets"]
        else:
            # 加载和处理数据
            categories = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]

            # 获取训练和测试数据
            newsgroups_train = fetch_20newsgroups(
                subset="train",
                remove=("headers", "footers", "quotes"),
                categories=categories,
            )
            newsgroups_test = fetch_20newsgroups(
                subset="test",
                remove=("headers", "footers", "quotes"),
                categories=categories,
            )

            # TF-IDF 向量化
            vectorizer = TfidfVectorizer(
                stop_words="english", min_df=0.001, max_df=0.20
            )

            # 转换训练数据和测试数据
            train_vectors = vectorizer.fit_transform(newsgroups_train.data)
            test_vectors = vectorizer.transform(newsgroups_test.data)

            # 转换为密集数组并确保类型是 float32
            train_data = train_vectors.todense().astype(np.float32)
            test_data = test_vectors.todense().astype(np.float32)
            train_targets = newsgroups_train.target.astype(np.float32)
            test_targets = newsgroups_test.target.astype(np.float32)

            # 缓存所有数据
            with open(self.cache_file, "wb") as f:
                pickle.dump(
                    {
                        "train_data": train_data,
                        "test_data": test_data,
                        "train_targets": train_targets,
                        "test_targets": test_targets,
                    },
                    f,
                )

            # 根据 is_train 选择相应的数据
            if self.is_train:
                data = train_data
                targets = train_targets
            else:
                data = test_data
                targets = test_targets

        # 转换为 torch tensors
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)

    def __getitem__(self, index):
        """Return data, target, and index as float tensors."""
        return self.data[index].float(), self.targets[index].float(), index

    def __len__(self):
        # is_train or not
        return len(self.targets)
