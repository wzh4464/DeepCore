import os
import unittest
import numpy as np
import logging
from deepcore.datasets.adult import AdultDataset
from deepcore.datasets.flipped_dataset import FlippedDataset


class TestFlippedDataset(unittest.TestCase):
    def setUp(self):
        # Setup logging
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)

        # Setup dataset
        self.data_path = "./data/adult/adult-training.csv"  # Adjust path as needed
        self.logger.info(f"Absolute path of the dataset: {os.path.abspath(self.data_path)}")
        self.adult_dataset = AdultDataset(self.data_path)

        # Setup parameters for flipped dataset
        self.num_samples = 1000
        self.num_scores = 100
        self.num_flip = 50
        self.seed = 42

        # Create indices for subset
        self.indices = np.random.choice(
            len(self.adult_dataset), self.num_samples, replace=False
        )

    def test_adult_flipping(self):
        # Create flipped dataset
        flipped_dataset = FlippedDataset(
            self.adult_dataset,
            self.indices,
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
        for idx in flipped_indices:
            original_target = self.adult_dataset.targets[idx]
            _, flipped_target, _ = flipped_dataset[list(self.indices).index(idx)]
            self.assertEqual(original_target.item(), 0)  # Original label
            self.assertEqual(flipped_target, 1)  # Flipped label


if __name__ == "__main__":
    unittest.main()
