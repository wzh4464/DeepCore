###
# File: ./tests/test_flipped_dataset.py
# Created Date: Monday, November 25th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 25th November 2024 7:58:27 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import unittest
import numpy as np
import logging
import torch
from deepcore.datasets.flipped_dataset import FlippedDataset
from deepcore.datasets.adult import Adult, AdultDataset
from deepcore.datasets.news import News20, NewsDataset


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup logging
        cls.logger = logging.getLogger("test_logger")
        cls.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        cls.logger.addHandler(handler)

        # Common test parameters
        cls.num_samples = 1000
        cls.num_scores = 100
        cls.num_flip = 50
        cls.seed = 42

    def setUp(self):
        """Setup test data for each test method"""
        # Setup paths
        self.data_dir = "./data"
        self.adult_path = os.path.join(self.data_dir, "adult")
        self.news_path = os.path.join(self.data_dir, "news")

        # Create directories if they don't exist
        os.makedirs(self.adult_path, exist_ok=True)
        os.makedirs(self.news_path, exist_ok=True)

        # Log paths
        self.logger.info(f"Adult data path: {os.path.abspath(self.adult_path)}")
        self.logger.info(f"News data path: {os.path.abspath(self.news_path)}")

    def test_adult_dataset_load(self):
        """Test loading of Adult dataset"""
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = (
            Adult(self.adult_path)
        )

        # Test basic properties
        self.assertEqual(channel, 1)
        self.assertIsNone(im_size)
        self.assertEqual(num_classes, 2)
        self.assertEqual(len(class_names), 2)
        self.assertIsNone(mean)
        self.assertIsNone(std)

        # Test dataset objects
        self.assertIsInstance(dst_train, AdultDataset)
        self.assertIsInstance(dst_test, AdultDataset)

        # Test data types and shapes
        x, y = dst_train[0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_news_dataset_load(self):
        """Test loading of News20 dataset"""
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = (
            News20(self.news_path)
        )

        # Test basic properties
        self.assertEqual(channel, 1)
        self.assertIsNone(im_size)
        self.assertEqual(num_classes, 2)
        self.assertEqual(len(class_names), 2)
        self.assertIsNone(mean)
        self.assertIsNone(std)

        # Test dataset objects
        self.assertIsInstance(dst_train, NewsDataset)
        self.assertIsInstance(dst_test, NewsDataset)

        # Test data types and shapes
        x, y = dst_train[0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_adult_flipping(self):
        """Test flipping functionality with Adult dataset"""
        # Load dataset
        _, _, _, _, _, _, dst_train, _ = Adult(self.adult_path)

        # Create indices for subset
        indices = np.random.choice(len(dst_train), self.num_samples, replace=False)

        # Create flipped dataset
        flipped_dataset = FlippedDataset(
            dst_train,
            indices,
            self.num_scores,
            self.num_flip,
            "adult",
            self.seed,
            self.logger,
        )

        # Test dataset size
        self.assertEqual(len(flipped_dataset), self.num_samples)

        # Test number of flipped samples
        flipped_indices = flipped_dataset.get_flipped_indices()
        self.assertEqual(len(flipped_indices), self.num_flip)

        # Test scores indices
        scores_indices = flipped_dataset.get_flipped_selection_from()
        self.assertEqual(len(scores_indices), self.num_scores)

        # Verify all flipped indices are included in scores indices
        flipped_set = set(flipped_indices)
        scores_set = set(scores_indices)
        self.assertTrue(flipped_set.issubset(scores_set))

        # Test label flipping
        # sourcery skip: no-loop-in-tests
        for idx in flipped_indices:
            original_target = dst_train.targets[idx]
            _, flipped_target, _ = flipped_dataset[list(indices).index(idx)]
            self.assertEqual(original_target.item(), 0)  # Original label should be 0
            self.assertEqual(flipped_target, 1)  # Flipped label should be 1

    def test_news_flipping(self):
        """Test flipping functionality with News20 dataset"""
        # Load dataset
        _, _, _, _, _, _, dst_train, _ = News20(self.news_path)

        # Create indices for subset
        indices = np.random.choice(len(dst_train), self.num_samples, replace=False)

        # Create flipped dataset
        flipped_dataset = FlippedDataset(
            dst_train,
            indices,
            self.num_scores,
            self.num_flip,
            "News20",
            self.seed,
            self.logger,
        )

        # Test dataset size
        self.assertEqual(len(flipped_dataset), self.num_samples)

        # Test number of flipped samples
        flipped_indices = flipped_dataset.get_flipped_indices()
        self.assertEqual(len(flipped_indices), self.num_flip)

        # Test scores indices
        scores_indices = flipped_dataset.get_flipped_selection_from()
        self.assertEqual(len(scores_indices), self.num_scores)

        # Verify all flipped indices are included in scores indices
        flipped_set = set(flipped_indices)
        scores_set = set(scores_indices)
        self.assertTrue(flipped_set.issubset(scores_set))

        # Test label flipping
        # sourcery skip: no-loop-in-tests
        for idx in flipped_indices:
            original_target = dst_train.targets[idx]
            _, flipped_target, _ = flipped_dataset[list(indices).index(idx)]
            self.assertEqual(original_target.item(), 0)  # Original label should be 0
            self.assertEqual(flipped_target, 1)  # Flipped label should be 1


if __name__ == "__main__":
    unittest.main()
