###
# File: ./liveval/datasets/cache_tils.py
# Created Date: Monday, December 9th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 27th January 2025 4:50:55 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import pickle
from filelock import FileLock
import logging


class DatasetCache:
    """Dataset caching utility class."""

    def __init__(self, cache_dir, logger=None):
        """
        Initialize the cache utility.

        Args:
            cache_dir (str): Directory to store cache files
            logger (logging.Logger, optional): Logger instance
        """
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, dataset_name, **kwargs):
        """Generate cache file path based on parameters."""
        # Create a unique cache key from the parameters
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return os.path.join(self.cache_dir, f"{dataset_name}_{param_str}.pkl")

    def load_or_create(self, cache_path, create_fn):
        """
        Load from cache if exists, otherwise create and cache.

        Args:
            cache_path (str): Path to cache file
            create_fn (callable): Function to create data if not cached

        Returns:
            The cached or newly created data
        """
        lock_path = f"{cache_path}.lock"

        with FileLock(lock_path):
            if os.path.exists(cache_path):
                self.logger.info(f"Loading from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            self.logger.info("Cache miss. Creating new data...")
            data = create_fn()

            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            self.logger.info(f"Data cached at: {cache_path}")
            return data
