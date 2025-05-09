class CoresetMethod(object):
    """
    A base class for coreset selection methods in deep learning.

    This class provides a general framework for implementing different coreset selection methods.
    It handles basic initialization, such as storing the training dataset, calculating the size
    of the coreset, and storing relevant arguments. Subclasses should override the `select` method
    to implement specific coreset selection logic.

    Attributes:
        dst_train (Dataset): The training dataset from which the coreset will be selected.
        args (argparse.Namespace): Arguments containing various settings and configurations for
                                    the coreset selection method.
        fraction (float): The fraction of the dataset to be selected as the coreset. Must be
                          between 0 and 1.
        random_seed (int, optional): The seed for random number generation to ensure reproducibility.
                                     Defaults to None.
        num_classes (int): The number of classes in the training dataset.
        index (list): A list to store the indices of the selected samples.
        n_train (int): The total number of samples in the training dataset.
        coreset_size (int): The number of samples to be selected for the coreset, computed as a
                            fraction of the total training dataset size.
    """

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        """
        Initializes the CoresetMethod with the training dataset and relevant parameters.

        Args:
            dst_train (Dataset): The training dataset from which the coreset will be selected.
            args (argparse.Namespace): Arguments containing various settings and configurations.
            fraction (float, optional): The fraction of the dataset to select as the coreset.
                                        Defaults to 0.5.
            random_seed (int, optional): The seed for random number generation to ensure reproducibility.
                                         Defaults to None.
            **kwargs: Additional keyword arguments for future extensions.

        Raises:
            ValueError: If the `fraction` is not between 0 and 1.
        """
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")

        self.dst_train = dst_train  # Save the training dataset
        # 兼容 Subset 类型，递归查找 .classes 属性
        base = dst_train
        while hasattr(base, 'dataset'):
            base = base.dataset
        if hasattr(base, 'classes'):
            self.num_classes = len(base.classes)
        else:
            raise AttributeError('Training dataset (or its base) must have a .classes attribute')
        self.fraction = fraction  # Save the fraction of the dataset to be selected
        self.random_seed = random_seed  # Save the seed value for random selection (for reproducibility)
        self.index = []  # List to store the indices of the selected samples
        self.args = (
            args  # Save the arguments object (may contain other configuration options)
        )

        self.n_train = len(
            dst_train
        )  # Get the total number of samples in the training dataset
        self.coreset_size = round(
            self.n_train * fraction
        )  # Calculate the number of samples to be selected based on the fraction

    def select(self, **kwargs):
        """
        The method to be overridden by subclasses to implement specific coreset selection logic.

        This method should return the indices of the selected samples from the training dataset.

        Args:
            **kwargs: Additional keyword arguments for specific selection methods.

        Returns:
            None: The method should return a list or array of selected sample indices in subclasses.
        """
        return

    def get_scores(self, **kwargs):
        """
        A generic method to calculate the scores for the selected samples.

        This method should be overridden by subclasses to implement specific scoring logic.

        Args:
            **kwargs: Additional keyword arguments for specific scoring methods.

        Returns:
            None: The method should return a list or array of scores for the selected samples.
        """
        return
