###
# File: ./liveval/methods/earlytrain.py
# Created Date: Wednesday, November 13th 2024
# Author: Zihan
# -----
# Last Modified: Monday, 12th May 2025 9:22:26 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms
import logging
from liveval.utils.exception_utils import log_exception, ExceptionHandler


class EarlyTrain(CoresetMethod):
    """
    Core code for training related to coreset selection methods when pre-training is required.

    This class provides a framework for implementing coreset selection methods that involve
    pre-training or early training stages. It handles the setup of the training environment,
    including model initialization, optimizer setup, and the training loop.

    Attributes:
        epochs (int): Number of epochs to train.
        n_train (int): Number of samples in the training dataset.
        coreset_size (int): Size of the coreset to be selected.
        specific_model (str): Name of a specific model to use, if any.
        fraction_pretrain (float): Fraction of the dataset to use for pretraining.
        dst_pretrain_dict (dict): Dictionary containing pretraining dataset information.
        torchvision_pretrain (bool): Whether to use pretrained models from torchvision.
        if_dst_pretrain (bool): Whether a separate pretraining dataset is provided.
        n_pretrain (int): Number of samples in the pretraining dataset.
        n_pretrain_size (int): Number of samples to use for pretraining.
        dst_test (Dataset): Test dataset, if provided.

    Args:
        dst_train (Dataset): The training dataset.
        args (argparse.Namespace): Arguments containing various settings.
        fraction (float, optional): Fraction of the dataset to select as coreset. Defaults to 0.5.
        random_seed (int, optional): Seed for random number generation. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        specific_model (str, optional): Name of a specific model to use. Defaults to None.
        torchvision_pretrain (bool, optional): Whether to use pretrained models from torchvision. Defaults to False.
        dst_pretrain_dict (dict, optional): Dictionary containing pretraining dataset information. Defaults to {}.
        fraction_pretrain (float, optional): Fraction of the dataset to use for pretraining. Defaults to 1.0.
        dst_test (Dataset, optional): Test dataset. Defaults to None.
    """

    @log_exception()
    def __init__(
        self,
        dst_train,
        args,
        fraction=0.5,
        random_seed=None,
        epochs=200,
        specific_model=None,
        torchvision_pretrain: bool = False,
        dst_pretrain_dict: dict = None,
        fraction_pretrain=1.0,
        dst_test=None,
        **kwargs,
    ):
        """
        Initialize the EarlyTrain instance.

        define:
        - self.epochs (int): Number of epochs to train. e.g. 5
        - self.n_train (int): Number of samples in the training dataset. e.g. 60000
        - self.coreset_size (int): Size of the coreset to be selected. e.g. 6000
        - self.specific_model (str): Name of a specific model to use, if any. e.g. None
        - self.fraction_pretrain (float): Fraction of the dataset to use for pretraining. e.g. 1.0
        - self.dst_pretrain_dict (dict): Dictionary containing pretraining dataset information. e.g. {}
        - self.torchvision_pretrain (bool): Whether to use pretrained models from torchvision. e.g. False
        - self.if_dst_pretrain (bool): Whether a separate pretraining dataset is provided. e.g. False
        - self.n_pretrain (int): Number of samples in the pretraining dataset. e.g. 60000
        - self.n_pretrain_size (int): Number of samples to use for pretraining. e.g. 60000
        - self.dst_test (Dataset): Test dataset, if provided. e.g. None
        - self.scheduler (None): e.g. COSineAnnealingLR
        - self.current_epoch (int): e.g. 0
        - self.current_step (int): e.g. 0

        Sets up the training environment, including dataset preparation, model selection,
        and optimization settings.
        """
        if dst_pretrain_dict is None:
            dst_pretrain_dict = {}
        super().__init__(dst_train, args, fraction, random_seed)
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"__init__({dst_train}, {args}, {fraction}, {random_seed}, {epochs}, {specific_model}, {torchvision_pretrain}, {dst_pretrain_dict}, {fraction_pretrain}, {dst_test}, {kwargs})"
        )
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if fraction_pretrain <= 0.0 or fraction_pretrain > 1.0:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if (
                "im_size" not in dict_keys
                or "channel" not in dict_keys
                or "dst_train" not in dict_keys
                or "num_classes" not in dict_keys
            ):
                raise AttributeError(
                    "Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes."
                )
            if dst_pretrain_dict["im_size"][0] != args.im_size[0]:
                raise ValueError(
                    "im_size of pretrain dataset does not match that of the training dataset."
                )
            if dst_pretrain_dict["channel"] != args.channel:
                raise ValueError(
                    "channel of pretrain dataset does not match that of the training dataset."
                )
            if dst_pretrain_dict["num_classes"] != args.num_classes:
                self.num_classes_mismatch()

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = len(self.dst_pretrain_dict) != 0

        if torchvision_pretrain and (args.im_size[0] != 224 or args.im_size[1] != 224):
            self.dst_train = deepcopy(dst_train)
            self.dst_train.transform = transforms.Compose(
                [self.dst_train.transform, transforms.Resize(224)]
            )
            if self.if_dst_pretrain:
                self.dst_pretrain_dict["dst_train"] = deepcopy(
                    dst_pretrain_dict["dst_train"]
                )
                self.dst_pretrain_dict["dst_train"].transform = transforms.Compose(
                    [
                        self.dst_pretrain_dict["dst_train"].transform,
                        transforms.Resize(224),
                    ]
                )
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict["dst_train"])
        self.n_pretrain_size = round(
            self.fraction_pretrain
            * (self.n_pretrain if self.if_dst_pretrain else self.n_train)
        )
        self.dst_test = dst_test
        # self.args.scheduler = "CosineAnnealingLR"
        self.scheduler = None
        self.current_epoch = 0
        self.current_step = 0

    def _set_random_seeds(self, seed):
        """
        Set all random seeds to ensure reproducibility.
        """
        import random
        import torch.backends.cudnn as cudnn

        # Python random
        random.seed(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Additional settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.logger.info(f"Set all random seeds to {seed}")

    @log_exception()
    def train(self, epoch, list_of_train_idx, **kwargs):
        """
        Train the model for one epoch.
        """
        self.logger.debug(f"train({epoch}, {list_of_train_idx}, {kwargs})")
        self.before_train()
        self.model.train()

        print("\n=> Training Epoch #%d" % epoch)

        # 选用原始数据集和 SubsetRandomSampler，避免索引错位
        if self.if_dst_pretrain:
            dst_data = self.dst_pretrain_dict["dst_train"]
        else:
            dst_data = self.dst_train
        # 如果是 Subset，取其原始数据集
        if hasattr(dst_data, 'indices') and hasattr(dst_data, 'dataset'):
            base_dataset = dst_data.dataset
        else:
            base_dataset = dst_data

        train_loader = torch.utils.data.DataLoader(
            base_dataset,
            batch_size=self.args.selection_batch,
            sampler=torch.utils.data.SubsetRandomSampler(list_of_train_idx),
            num_workers=self.args.workers,
            pin_memory=True,
        )

        for i, (inputs, targets, true_indices) in enumerate(train_loader):
            self.logger.info(f"Training batch {i} of {len(train_loader)}")
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 这里 trainset_permutation_inds[i] 可能不再适用，直接传 true_indices
            self.after_loss(outputs, loss, targets, true_indices, epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(
                outputs, loss, targets, epoch, i, self.args.selection_batch
            )

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    @log_exception()
    def run(self):
        """
        Run the entire training process.

        This method sets up the model, optimizer, and runs the training loop for the specified number of epochs.

        Returns:
            The result of finish_run method.
        """
        self.logger.info("run()")

        for epoch in range(self.epochs):
            dst_train = (
                self.dst_pretrain_dict["dst_train"]
                if self.if_dst_pretrain
                else self.dst_train
            )
            if hasattr(dst_train, "indices"):
                list_of_train_idx = dst_train.indices.tolist()
            else:
                list_of_train_idx = list(range(len(dst_train)))
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            if (
                self.dst_test is not None
                and self.args.selection_test_interval > 0
                and (epoch + 1) % self.args.selection_test_interval == 0
            ):
                self.test(epoch)
            self.after_epoch()

        return self.finish_run()

    @log_exception()
    def setup_optimizer_and_scheduler(self):
        """
        Defined:
        - self.model_optimizer: Optimizer to use.
        - self.scheduler: Scheduler to use.
        """
        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.model_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.selection_lr,
                momentum=self.args.selection_momentum,
                weight_decay=self.args.selection_weight_decay,
                nesterov=self.args.selection_nesterov,
            )
        elif self.args.selection_optimizer == "Adam":
            self.model_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.selection_lr,
                weight_decay=self.args.selection_weight_decay,
            )
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](
                self.model.parameters(),
                lr=self.args.selection_lr,
                momentum=self.args.selection_momentum,
                weight_decay=self.args.selection_weight_decay,
                nesterov=self.args.selection_nesterov,
            )

        # Setup scheduler
        if self.args.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optimizer, T_max=self.epochs
            )
        elif self.args.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.model_optimizer, step_size=30, gamma=0.1
            )
        # Add more scheduler options as needed

    @log_exception()
    def test(self, epoch):
        """
        Test the model's performance.

        NOTE TO DEVELOPERS: 
        If you override this `test` method in a subclass, ensure you correctly 
        initialize the `test_loader` using `self._get_test_loader()` and handle 
        the case where it might be `None`, similar to the implementation below.
        This ensures consistent and safe test data loading.

        Args:
            epoch (int): Current epoch number.
        """
        if self.dst_test is None:
            self.logger.error("No test dataset provided (self.dst_test is None). Skipping test.")
            return
        self.logger.info(f"test({epoch})")
        self.model.no_grad = True
        self.model.eval()

        test_loader, _ = self._get_test_loader()
        if test_loader is None:
            return

        correct = 0.0
        total = 0.0

        print("\n=> Testing Epoch #%d" % epoch)

        for batch_idx, (input, target, _) in enumerate(test_loader):
            output = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                self.logger.info(
                    f"| Test Epoch [{epoch}/{self.epochs}] Iter[{batch_idx + 1}/{len(test_loader)}]\t\tTest Loss: {loss.item()} Test Acc: {100.0 * correct / total}%"
                )
                self.logger.debug(f"Correct: {correct}, Total: {total}")

        self.model.no_grad = False
        self.last_test_acc = correct / total if total > 0 else 0.0

    def num_classes_mismatch(self):
        """
        Handle mismatch in number of classes between pretraining and training datasets.
        """
        pass

    def before_train(self):
        """
        Perform actions before training starts.
        """
        self.logger.info("before_train()")

    def get_lr(self):
        return self.model_optimizer.param_groups[0]["lr"]

    def step_scheduler(self):
        if self.scheduler:
            self.scheduler.step()
            print(f"[EarlyTrain] Stepped scheduler. New LR: {self.get_lr()}")

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        """
        Perform actions after loss computation.

        Args:
            outputs: Model outputs.
            loss: Computed loss.
            targets: Ground truth labels.
            batch_inds: Indices of the current batch.
            epoch: Current epoch number.
        """
        self.logger.debug(
            f"after_loss({outputs}, {loss}, {targets}, {batch_inds}, {epoch})"
        )

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        """
        Perform actions during the update step.

        Args:
            outputs: Model outputs.
            loss: Computed loss.
            targets: Ground truth labels.
            epoch: Current epoch number.
            batch_idx: Current batch index.
            batch_size: Size of the current batch.
        """
        self.logger.debug(
            f"while_update({outputs}, {loss}, {targets}, {epoch}, {batch_idx}, {batch_size})"
        )

    def finish_train(self):
        """
        Perform actions after training is finished.

        Returns:
            Any result from the training process.
        """
        self.logger.info("finish_train()")

    @log_exception()
    def before_epoch(self):
        """
        Perform actions before each epoch starts.
        """
        self.logger.info("before_epoch()")

    @log_exception()
    def after_epoch(self):
        """
        Perform actions after each epoch ends.
        """
        self.logger.info("after_epoch()")

    @log_exception()
    def before_run(self):
        """
        Perform actions before the entire run starts.

        Defined:
        - self.model: Model to use for training.
        - self.criterion: Loss function to use.
        - self.train_indx: Indices of the training samples.
        - self.model_optimizer: Optimizer to use.
        - self.current_epoch: Current epoch number.
        - self.current_step: Current step number.
        """
        self.logger.info("before_run()")
        self._set_random_seeds(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        model_name = (
            self.args.model if self.specific_model is None else self.specific_model
        )

        # Check if the model exists in nets.__dict__ before accessing
        if model_name in nets.__dict__:
            self.model = nets.__dict__[model_name](
                self.args.channel,
                (
                    self.dst_pretrain_dict["num_classes"]
                    if self.if_dst_pretrain
                    else self.num_classes
                ),
                pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size,
            ).to(self.args.device)
        else:
            raise ValueError(f"Model '{model_name}' not found in nets.__dict__")

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.device == "mps":
            print("Using MPS.")
            self.model = self.model.to("mps")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.model = nets.nets_utils.MyDataParallel(
                self.model, device_ids=self.args.gpu
            )
        elif torch.cuda.device_count() > 1:
            self.model = nets.nets_utils.MyDataParallel(self.model).cuda()

        # 在 before_run 方法中
        if self.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss().to(self.args.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()

    def finish_run(self):
        """
        Perform actions after the entire run is finished.

        Returns:
            Any result from the run process.
        """
        self.logger.info("finish_run()")
        return

    def _initialize_data_loader(self):
        """
        Initializes the data loader for training.
        Make sure:
        - self.train_loader is initialized with the training data.
        - self.train_indices is initialized with the indices of the training data.
        - self.train_iterator is initialized with the iterator for the training loader.
        """
        if not hasattr(self, "train_loader") or not hasattr(self, "train_iterator"):
            self.train_loader, self.train_indices = self._get_train_loader()
            self.train_iterator = iter(self.train_loader)

    def select(self, **kwargs):
        """
        Perform the selection process.

        Returns:
            selection_result: Result of the selection process.
        """
        self.logger.info(f"select({kwargs})")
        self.before_run()
        return self.run()

    def _get_train_loader(self):
        """
        Create and return training data loader.
        
        Returns:
            tuple: (train_loader, train_indices)
                - train_loader: DataLoader for training data
                - train_indices: Indices of the training data
        """
        self.logger.info("Creating training data loader.")
        
        # Create DataLoader
        train_loader = torch.utils.data.DataLoader(
            self.dst_train,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )
        
        self.logger.info(
            "Training data loader created with batch size %d and %d workers.",
            self.args.selection_batch,
            self.args.workers,
        )
        
        return train_loader, np.arange(len(self.dst_train))

    @log_exception()
    def train_for_epochs(self, num_epochs, train_indices, test_loader=None):
        """
        训练模型指定轮数，并在每轮后可选地进行测试。

        参数：
            num_epochs (int): 训练的轮数。
            train_indices (list): 训练样本的索引列表。
            test_loader (DataLoader, optional): 测试数据加载器。默认为 None。
        返回：
            float: 如果提供 test_loader，返回训练后最终的测试准确率。
        """
        self.logger.info(f"Training for {num_epochs} epochs with {len(train_indices)} samples")

        # 确保模型已初始化
        if not hasattr(self, 'model') or self.model is None:
            self.before_run()

        test_acc = None

        for epoch in range(num_epochs):
            self.before_epoch()
            self.train(epoch, train_indices)

            # 如有需要，进行测试
            if test_loader is not None:
                _ = self.test(epoch)  # 保持与原有 test 方法兼容
            self.after_epoch()

        # 最终评估
        if test_loader is not None:
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets, _ in test_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = correct / total if total > 0 else 0.0
            self.last_test_acc = test_acc
            self.logger.info(f"Final test accuracy: {test_acc:.4f}")
        return test_acc

    def _get_test_loader(self):
        """
        创建并返回测试数据的 DataLoader。
        保证 self.dst_test 不为 None。
        返回：
            tuple: (test_loader, test_indices)
                - test_loader: DataLoader for test data
                - test_indices: Indices of the test data
        """
        if self.dst_test is None:
            self.logger.error("No test dataset provided (self.dst_test is None). Skipping test.")
            return None, None
        test_loader = torch.utils.data.DataLoader(
            self.dst_test,
            batch_size=self.args.selection_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )
        indices = np.arange(len(self.dst_test))
        return test_loader, indices
