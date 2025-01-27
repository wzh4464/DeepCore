###
# File: ./main.py
# Created Date: Monday, October 21st 2024
# Author: Zihan
# -----
# Last Modified: Monday, 27th January 2025 4:50:15 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import multiprocessing
import os
import torch.nn as nn
import argparse
import liveval.nets as nets
import liveval.datasets as datasets
from liveval.datasets.flipped_dataset import FlippedDataset
import liveval.methods as methods
from liveval.datasets.corrupted_dataset import CorruptedDataset
from liveval.methods.selection_methods import SELECTION_METHODS
from liveval.methods.coresetmethod import CoresetMethod
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
from typing import Type
import logging


def parse_args():
    """
    Parse command line arguments for the DeepCore project.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    Command line arguments:
        --dataset (str): Dataset to use (default: "CIFAR10").
        --model (str): Model to use (default: "ResNet18").
        --selection (str): Selection method (default: "uniform").
        --num_exp (int): Number of experiments (default: 5).
        --num_eval (int): Number of evaluating randomly initialized models (default: 10).
        --epochs (int): Number of total epochs to run (default: 200).
        --data_path (str): Dataset path (default: "data").
        --gpu (int): GPU id to use (default: None).
        --print_freq (int): Print frequency (default: 20).
        --fraction (float): Fraction of data to be selected (default: 0.1).
        --seed (int): Random seed (default: current time in milliseconds % 100000).
        --workers (int): Number of data loading workers (default: 8).
        --cross (str): Models for cross-architecture experiments (default: None).
        --optimizer (str): Optimizer to use (default: "SGD").
        --lr (float): Learning rate for updating network parameters (default: 0.1).
        --min_lr (float): Minimum learning rate (default: 1e-4).
        --momentum (float): Momentum (default: 0.9).
        --weight_decay (float): Weight decay (default: 5e-4).
        --nesterov (bool): If set, use Nesterov momentum (default: True).
        --scheduler (str): Learning rate scheduler (default: "CosineAnnealingLR").
        --gamma (float): Gamma value for StepLR (default: 0.5).
        --step_size (float): Step size for StepLR (default: 50).
        --batch (int): Mini-batch size (default: 256).
        --train_batch (int): Batch size for training (default: None).
        --selection_batch (int): Batch size for selection (default: None).
        --test_interval (int): Number of training epochs between two test epochs (default: 1).
        --test_fraction (float): Proportion of test dataset used for evaluating the model (default: 1.0).
        --selection_epochs (int): Number of epochs while performing selection on full dataset (default: 40).
        --selection_momentum (float): Momentum while performing selection (default: 0.9).
        --selection_weight_decay (float): Weight decay while performing selection (default: 5e-4).
        --selection_optimizer (str): Optimizer to use while performing selection (default: "SGD").
        --selection_nesterov (bool): If set, use Nesterov momentum while performing selection (default: True).
        --selection_lr (float): Learning rate for selection (default: 0.1).
        --selection_test_interval (int): Number of training epochs between two test epochs during selection (default: 1).
        --selection_test_fraction (float): Proportion of test dataset used for evaluating the model while performing selection (default: 1.0).
        --balance (bool): Whether balance selection is performed per class (default: True).
        --submodular (str): Submodular function to use (default: "GraphCut").
        --submodular_greedy (str): Greedy algorithm for submodular optimization (default: "LazyGreedy").
        --uncertainty (str): Uncertainty score to use (default: "Entropy").
        --save_path (str): Path to save results (default: "").
        --resume (str): Path to latest checkpoint (default: "").
        --num_gpus (int): Number of GPUs to use for OTI (default: 3).
        --oti_mode (str): OTI operation mode (default: "scores").
        --oti_use_regularization (bool): Use regularization in OTI score calculation.
        --oti_use_learning_rate (bool): Use learning rate in OTI score calculation.
        --oti_use_sliding_window (bool): Use sliding window in OTI score calculation.
        --eps_min (float): Minimum threshold for loss change (default: 0.1).
        --eps_max (float): Maximum threshold for loss change (default: 0.05).
        --delta_min (float): Minimum threshold for parameter change (default: 0.1).
        --delta_max (float): Maximum threshold for parameter change (default: 0.05).
        --delta_step (float): Step size for parameter change (default: 0.01).
        --log_level (str): Set the logging level (default: "INFO").
        --num_scores (int): Number of scores to calculate for LOO (default: 100).
        --exp (str): Specify the experiment mode: 'train_and_eval' or 'flip'. Default is 'train_and_eval'.
        --num_flip (int): Number of flips to perform in 'flip' mode (default
    """
    parser = argparse.ArgumentParser(description="Parameter Processing")

    # Basic arguments
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset")
    parser.add_argument("--model", type=str, default="ResNet18", help="model")
    parser.add_argument(
        "--selection", type=str, default="uniform", help="selection method"
    )
    parser.add_argument(
        "--num_exp", type=int, default=5, help="the number of experiments"
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=10,
        help="the number of evaluating randomly initialized models",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--data_path", type=str, default="data", help="dataset path")
    parser.add_argument(
        "--gpu", default=None, nargs="+", type=int, help="GPU id to use"
    )
    parser.add_argument(
        "--print_freq", "-p", default=20, type=int, help="print frequency (default: 20)"
    )
    parser.add_argument(
        "--fraction",
        default=0.1,
        type=float,
        help="fraction of data to be selected (default: 0.1)",
    )
    parser.add_argument(
        "--seed", default=int(time.time() * 1000) % 100000, type=int, help="random seed"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--cross",
        type=str,
        nargs="+",
        default=None,
        help="models for cross-architecture experiments",
    )

    # Optimizer and scheduler
    parser.add_argument(
        "--optimizer", default="SGD", help="optimizer to use, e.g. SGD, Adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate for updating network parameters",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-4, help="minimum learning rate"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 5e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--nesterov", default=True, type=str_to_bool, help="if set nesterov"
    )
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        type=str,
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma value for StepLR"
    )
    parser.add_argument(
        "--step_size", type=float, default=50, help="Step size for StepLR"
    )

    # Training
    parser.add_argument(
        "--batch",
        "--batch-size",
        "-b",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--train_batch",
        "-tb",
        default=None,
        type=int,
        help="batch size for training, if not specified, it will equal to batch size in argument --batch",
    )
    parser.add_argument(
        "--selection_batch",
        "-sb",
        default=None,
        type=int,
        help="batch size for selection, if not specified, it will equal to batch size in argument --batch",
    )

    # Testing
    parser.add_argument(
        "--test_interval",
        "-ti",
        default=1,
        type=int,
        help="the number of training epochs to be performed between two test epochs; a value of 0 means no test will be run (default: 1)",
    )
    parser.add_argument(
        "--test_fraction",
        "-tf",
        type=float,
        default=1.0,
        help="proportion of test dataset used for evaluating the model (default: 1.)",
    )

    # Selecting
    parser.add_argument(
        "--selection_epochs",
        "-se",
        default=40,
        type=int,
        help="number of epochs while performing selection on full dataset",
    )
    parser.add_argument(
        "--selection_momentum",
        "-sm",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum while performing selection (default: 0.9)",
    )
    parser.add_argument(
        "--selection_weight_decay",
        "-swd",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay while performing selection (default: 5e-4)",
        dest="selection_weight_decay",
    )
    parser.add_argument(
        "--selection_optimizer",
        "-so",
        default="SGD",
        help="optimizer to use while performing selection, e.g. SGD, Adam",
    )
    parser.add_argument(
        "--selection_nesterov",
        "-sn",
        default=True,
        type=str_to_bool,
        help="if set nesterov while performing selection",
    )
    parser.add_argument(
        "--selection_lr",
        "-slr",
        type=float,
        default=0.1,
        help="learning rate for selection",
    )
    parser.add_argument(
        "--selection_test_interval",
        "-sti",
        default=1,
        type=int,
        help="the number of training epochs to be performed between two test epochs during selection (default: 1)",
    )
    parser.add_argument(
        "--selection_test_fraction",
        "-stf",
        type=float,
        default=1.0,
        help="proportion of test dataset used for evaluating the model while performing selection (default: 1.)",
    )
    parser.add_argument(
        "--balance",
        default=True,
        type=str_to_bool,
        help="whether balance selection is performed per class",
    )

    # Algorithm
    parser.add_argument(
        "--submodular", default="GraphCut", help="specify submodular function to use"
    )
    parser.add_argument(
        "--submodular_greedy",
        default="LazyGreedy",
        help="specify greedy algorithm for submodular optimization",
    )
    parser.add_argument(
        "--uncertainty", default="Entropy", help="specify uncertainty score to use"
    )

    # Checkpoint and resumption
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default="",
        help="path to save results (default: do not save)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default="",
        help="path to latest checkpoint (default: do not load)",
    )

    # OTI specific arguments
    parser.add_argument(
        "--num_gpus", type=int, default=3, help="number of GPUs to use for OTI"
    )
    parser.add_argument(
        "--oti_mode",
        type=str,
        default="scores",
        choices=["full", "stored", "scores"],
        help="OTI operation mode",
    )
    parser.add_argument(
        "--oti_use_regularization",
        action="store_true",
        help="Use regularization in OTI score calculation",
    )
    parser.add_argument(
        "--oti_use_learning_rate",
        action="store_true",
        help="Use learning rate in OTI score calculation",
    )
    parser.add_argument(
        "--oti_use_sliding_window",
        action="store_true",
        help="Use sliding window in OTI score calculation",
    )

    # 新增 eps_min 和 eps_max delta_min delta_max delta_step
    parser.add_argument(
        "--eps_min",
        type=float,
        default=0.1,
        help="Minimum threshold for loss change",
    )
    parser.add_argument(
        "--eps_max",
        type=float,
        default=0.05,
        help="Maximum threshold for loss change",
    )
    parser.add_argument(
        "--delta_min",
        type=float,
        default=0.1,
        help="Minimum threshold for parameter change",
    )
    parser.add_argument(
        "--delta_max",
        type=float,
        default=0.05,
        help="Maximum threshold for parameter change",
    )
    parser.add_argument(
        "--delta_step",
        type=float,
        default=0.01,
        help="Step size for parameter change",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="train_and_eval",
        choices=["train_and_eval", "flip", "corrupt"],
        help="Specify the experiment mode: 'train_and_eval', 'flip', or 'corrupt'. Default is 'train_and_eval'.",
    )

    parser.add_argument(
        "--num_corrupt",
        type=int,
        default=100,
        help="Number of inputs to corrupt in 'corrupt' mode (default: 100).",
    )

    parser.add_argument(
        "--num_flip",
        type=int,
        default=100,
        help="Number of flips to perform in 'flip' mode.",
    )

    # loo
    parser.add_argument(
        "--num_scores",
        type=int,
        default=100,
        help="Number of scores to calculate for LOO",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # add args.timestamp
    args.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return args


def setup_experiment(args):
    """
    Set up directories, checkpoint, and batch sizes.

    This function modifies the `args` object to ensure that batch sizes and directories
    are correctly set up. These modifications will affect subsequent processing,
    particularly in the `run_experiment` function.

    Modifications include:
    - Setting `args.train_batch` and `args.selection_batch` if they are not already specified.
    - Creating directories specified by `args.save_path` and `args.data_path` if they do not exist.
    - Loading the checkpoint if `args.resume` is provided, which updates `args` with the checkpoint details.

    Args:
        args: The argument namespace parsed from the command line.

    Returns:
        tuple: The loaded checkpoint dictionary (or an empty dict if not loading),
               the starting experiment number, and the starting epoch number.
    """
    logger = logging.getLogger(__name__)

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        logger.info(f"Created directory: {args.save_path}")
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
        logger.info(f"Created directory: {args.data_path}")

    if args.resume != "":
        checkpoint, start_exp, start_epoch = load_checkpoint(args)
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0
        logger.info("Starting new experiment")

    return checkpoint, start_exp, start_epoch


def load_checkpoint(args):
    """Load checkpoint if resume is specified."""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=args.device)
        assert {
            "exp",
            "epoch",
            "state_dict",
            "opt_dict",
            "best_acc1",
            "rec",
            "subset",
            "sel_args",
        } <= set(checkpoint.keys())
        assert "indices" in checkpoint["subset"].keys()
        start_exp = checkpoint["exp"]
        start_epoch = checkpoint["epoch"]
        logger.info(
            f"Checkpoint loaded. Resuming from experiment {start_exp}, epoch {start_epoch}"
        )
    except AssertionError:
        try:
            assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
            assert "indices" in checkpoint["subset"].keys()
            logger.info(
                "The checkpoint only contains the subset, training will start from the beginning"
            )
            start_exp = checkpoint["exp"]
            start_epoch = 0
        except AssertionError:
            logger.warning(
                "Failed to load the checkpoint, an empty one will be created"
            )
            checkpoint = {}
            start_exp = 0
            start_epoch = 0

    return checkpoint, start_exp, start_epoch


def initialize_dataset_and_model(args, checkpoint):
    """Initialize the dataset and model for training, including data loaders for training and testing."""
    logger, mean, std, dst_train, dst_test = initialize_logging_and_datasets(args)

    # Configure selection method and subset (if applicable)
    if "subset" in checkpoint.keys():
        subset = checkpoint["subset"]
        selection_args = checkpoint["sel_args"]
    else:
        selection_args = dict(
            epochs=args.selection_epochs,
            selection_method=args.uncertainty,
            balance=args.balance,
            greedy=args.submodular_greedy,
            function=args.submodular,
        )

        method_class = SELECTION_METHODS.get(args.selection)
        if method_class is None:
            raise ValueError(f"Selection method {args.selection} not found.")

        # Initialize selection method with specific OTI options if selected
        if args.selection == "OTI":
            method = method_class(
                dst_train,
                args,
                args.fraction,
                args.seed,
                num_gpus=args.num_gpus,
                mode=args.oti_mode,
                use_regularization=args.oti_use_regularization,
                use_learning_rate=args.oti_use_learning_rate,
                use_sliding_window=args.oti_use_sliding_window,
                dst_test=dst_test,
                **selection_args,
            )
        elif args.selection == "AD_OTI":
            method = method_class(
                dst_train,
                args,
                args.fraction,
                args.seed,
                num_gpus=args.num_gpus,
                mode=args.oti_mode,
                use_regularization=args.oti_use_regularization,
                use_learning_rate=args.oti_use_learning_rate,
                use_sliding_window=args.oti_use_sliding_window,
                dst_test=dst_test,
                eps_min=args.eps_min,
                eps_max=args.eps_max,
                delta_min=args.delta_min,
                delta_max=args.delta_max,
                delta_step=args.delta_step,
                **selection_args,
            )
        else:
            method = method_class(
                dst_train,
                args,
                args.fraction,
                args.seed,
                dst_test=dst_test,
                **selection_args,
            )

        subset = method.select()

    # Define data augmentation and preprocessing
    if args.dataset in ["CIFAR10", "CIFAR100"]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif args.dataset == "ImageNet":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    # Apply transforms to training dataset
    dst_train.transform = transform_train

    # Create subset if necessary (e.g., for OTI)
    if_weighted = "weights" in subset.keys()
    dst_subset = (
        WeightedSubset(dst_train, subset["indices"], subset["weights"])
        if if_weighted
        else torch.utils.data.Subset(dst_train, subset["indices"])
    )

    # Configure DataLoaders
    train_loader = torch.utils.data.DataLoader(
        dst_subset,
        batch_size=(args.train_batch),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dst_test,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    logger.info(f"train_loader_len: {len(train_loader)}")
    logger.info(f"test_loader_len: {len(test_loader)}")

    return train_loader, test_loader, if_weighted, subset, selection_args


def initialize_logging_and_datasets(args):
    logger = logging.getLogger(__name__)

    # Load dataset and basic dataset properties
    mean, std, dst_train, dst_test = initialize_dataset_properties(args)

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    return logger, mean, std, dst_train, dst_test


def initialize_dataset_properties(args):
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = (
        datasets.__dict__[args.dataset](args.data_path)
    )
    args.channel, args.im_size, args.num_classes, args.class_names = (
        channel,
        im_size,
        num_classes,
        class_names,
    )

    return mean, std, dst_train, dst_test


def initialize_flip_exp(args, seed):
    """
    初始化翻转实验。

    返回:
        flipped_train_loader: 包含部分翻转样本的训练集加载器。
        test_loader: 测试集加载器。
        flipped_set: 被翻转的样本索引。
        flipped_selection_set: 用于选择方法的样本索引。
        flipped_selection_args: 选择方法的相关参数。
    """
    logger, mean, std, dst_train, dst_test = initialize_logging_and_datasets(args)
    # dst_train, dst_test: 来自 torchvision 的数据集（例如 MNIST），未进行打乱。

    permuted_indices_path = os.path.join(args.save_path, "permuted_indices.csv")

    if os.path.exists(permuted_indices_path):
        permuted_indices = np.loadtxt(permuted_indices_path, delimiter=",").astype(int)
        logger.info(f"Loaded permuted indices from {permuted_indices_path}")
    else:
        # 使用 args.seed 和 args.num_scores 初始化 flipped_selection_set
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_indices = (
            torch.randperm(len(dst_train), generator=generator).numpy().astype(int)
        )

        # save permuted indices to csv
        np.savetxt(permuted_indices_path, permuted_indices, delimiter=",", fmt="%d")
        logger.info(f"Saved permuted indices to {permuted_indices_path}")

    # 创建包含部分翻转样本的训练集
    if args.exp == "flip":
        flipped_dataset = FlippedDataset(
            dst_train,
            permuted_indices,
            args.num_scores,
            args.num_flip,
            args.dataset,
            args.seed,
            logger,
        )
    else:
        flipped_dataset = CorruptedDataset(
            dataset=dst_train,
            indices=permuted_indices,
            num_scores=args.num_scores,
            num_corrupt=args.num_corrupt,
            dataset_name=args.dataset,
            seed=args.seed,
            logger=logger,
        )

    flipped_indices = flipped_dataset.get_flipped_indices()
    flipped_selection_from = flipped_dataset.get_flipped_selection_from()

    # save flipped indices to csv
    flipped_indices_path = os.path.join(args.save_path, "flipped_indices.csv")
    np.savetxt(flipped_indices_path, flipped_indices, delimiter=",", fmt="%d")
    logger.info(f"Saved flipped indices to {flipped_indices_path}")

    # flipped_train_loader = torch.utils.data.DataLoader(
    #     flipped_dataset,
    #     batch_size=args.selection_batch,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )

    # 创建测试集加载器
    test_loader = torch.utils.data.DataLoader(
        dst_test,
        batch_size=args.selection_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 创建 permutate 后 df 并保存. ind, label, flipped
    # df = pd.DataFrame(
    #     {
    #         "ind": [flipped_dataset[i][2] for i in range(len(flipped_dataset))],
    #         "label": [flipped_dataset[i][1] for i in range(len(flipped_dataset))],
    #         "flipped": [i in flipped_indices for i in permuted_indices],
    #     }
    # )

    # df.to_csv(os.path.join(args.save_path, "permuted_df.csv"), index=False)
    # logger.info(f"Saved permuted df to {os.path.join(args.save_path, 'permuted_df.csv')}")

    # logger.info("Initialize flip experiment successfully.")

    return (
        flipped_dataset,
        test_loader,
        flipped_indices,
        permuted_indices,
        flipped_selection_from,
    )


def initialize_network(args, model, train_loader, checkpoint, start_epoch):
    """Initialize the network, optimizer, and scheduler."""
    network = nets.__dict__[model](args.channel, args.num_classes, args.im_size).to(
        args.device
    )

    if args.device == "cpu":
        logging.warning("Using CPU.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()

    if "state_dict" in checkpoint.keys():
        network.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)

    optimizer = (
        torch.optim.SGD(
            network.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        if args.optimizer == "SGD"
        else (
            torch.optim.Adam(
                network.parameters(), args.lr, weight_decay=args.weight_decay
            )
            if args.optimizer == "Adam"
            else torch.optim.__dict__[args.optimizer](
                network.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(train_loader) * args.epochs, eta_min=args.min_lr
        )
        if args.scheduler == "CosineAnnealingLR"
        else (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=len(train_loader) * args.step_size,
                gamma=args.gamma,
            )
            if args.scheduler == "StepLR"
            else torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
        )
    )
    scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

    if "opt_dict" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["opt_dict"])

    return network, criterion, optimizer, scheduler


def print_experiment_info(args, exp, checkpoint_name):
    """Print the experiment information."""
    logger = logging.getLogger(__name__)

    logger.info(f"================== Exp {exp} ==================")
    logger.info(
        f"exp:{args.exp}, dataset: {args.dataset}, model: {args.model}, selection: {args.selection}, num_ex: {args.num_exp}, epochs: {args.epochs}, fraction: {args.fraction}, seed: {args.seed}, lr: {args.lr}, save_path: {args.save_path}, resume: {args.resume}, device: {args.device}, {'checkpoint_name: {checkpoint_name}' if args.save_path != '' else ''}"
    )


def train_and_evaluate_model(
    args,
    exp,
    start_epoch,
    train_loader,
    test_loader,
    subset,
    selection_args,
    checkpoint_name,
    model,
    checkpoint,
):
    """Train and evaluate a single model."""
    network, criterion, optimizer, scheduler = initialize_network(
        args, model, train_loader, checkpoint, start_epoch
    )
    rec = checkpoint["rec"] if "rec" in checkpoint.keys() else init_recorder()
    best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

    if args.save_path != "" and args.resume == "":
        save_checkpoint(
            {"exp": exp, "subset": subset, "sel_args": selection_args},
            os.path.join(
                args.save_path,
                checkpoint_name
                + ("" if model == args.model else f"{model}_")
                + "unknown.ckpt",
            ),
            0,
            0.0,
        )

    for epoch in range(start_epoch, args.epochs):
        train(
            train_loader,
            network,
            criterion,
            optimizer,
            scheduler,
            epoch,
            args,
            rec,
            if_weighted="weights" in subset,
        )

        if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
            prec1 = test(test_loader, network, criterion, epoch, args, rec)
            best_prec1 = save_best_checkpoint(
                args,
                exp,
                epoch,
                network,
                optimizer,
                best_prec1,
                prec1,
                rec,
                checkpoint_name,
                subset,
                selection_args,
                model,
            )

    finalize_checkpoint(
        args,
        exp,
        best_prec1,
        checkpoint_name,
        model,
        network,
        optimizer,
        rec,
        subset,
        selection_args,
    )


def save_best_checkpoint(
    args,
    exp,
    epoch,
    network,
    optimizer,
    best_prec1,
    prec1,
    rec,
    checkpoint_name,
    subset,
    selection_args,
    model,
):
    """Save the checkpoint if the current model has the best accuracy."""
    is_best = prec1 > best_prec1
    if is_best:
        best_prec1 = prec1
        if args.save_path != "":
            rec = record_ckpt(rec, epoch)
            save_checkpoint(
                {
                    "exp": exp,
                    "epoch": epoch + 1,
                    "state_dict": network.state_dict(),
                    "opt_dict": optimizer.state_dict(),
                    "best_acc1": best_prec1,
                    "rec": rec,
                    "subset": subset,
                    "sel_args": selection_args,
                },
                os.path.join(
                    args.save_path,
                    checkpoint_name
                    + ("" if model == args.model else f"{model}_")
                    + "unknown.ckpt",
                ),
                epoch=epoch,
                prec=best_prec1,
            )
    return best_prec1


def finalize_checkpoint(
    args,
    exp,
    best_prec1,
    checkpoint_name,
    model,
    network,
    optimizer,
    rec,
    subset,
    selection_args,
):
    """Finalize the checkpoint: rename or save the final checkpoint."""
    if args.save_path != "":
        try:
            os.rename(
                os.path.join(
                    args.save_path,
                    checkpoint_name
                    + ("" if model == args.model else f"{model}_")
                    + "unknown.ckpt",
                ),
                os.path.join(
                    args.save_path,
                    checkpoint_name
                    + ("" if model == args.model else f"{model}_")
                    + "%f.ckpt" % best_prec1,
                ),
            )
        except Exception:
            save_checkpoint(
                {
                    "exp": exp,
                    "epoch": args.epochs,
                    "state_dict": network.state_dict(),
                    "opt_dict": optimizer.state_dict(),
                    "best_acc1": best_prec1,
                    "rec": rec,
                    "subset": subset,
                    "sel_args": selection_args,
                },
                os.path.join(
                    args.save_path,
                    checkpoint_name
                    + ("" if model == args.model else f"{model}_")
                    + "%f.ckpt" % best_prec1,
                ),
                epoch=args.epochs - 1,
                prec=best_prec1,
            )


def run_experiment(args, checkpoint, start_exp, start_epoch):
    """Run the main training and evaluation loop."""
    logger = logging.getLogger(__name__)
    if args.exp == "train_and_eval":
        for exp in range(start_exp, args.num_exp):
            checkpoint_name = (
                setup_checkpoint_name(args, exp) if args.save_path != "" else ""
            )
            print_experiment_info(args, exp, checkpoint_name)

            train_loader, test_loader, if_weighted, subset, selection_args = (
                initialize_dataset_and_model(args, checkpoint)
            )
            models = [args.model] + (
                [model for model in args.cross if model != args.model]
                if isinstance(args.cross, list)
                else []
            )

            for model in models:
                if len(models) > 1:
                    logger.info(f"| Training on model {model}")
                train_and_evaluate_model(
                    args,
                    exp,
                    start_epoch,
                    train_loader,
                    test_loader,
                    subset,
                    selection_args,
                    checkpoint_name,
                    model,
                    checkpoint,
                )

            start_epoch = 0
            checkpoint = {}
            sleep(2)
    elif args.exp == "flip" or args.exp == "corrupt":
        _export_flipped_scores_summary(logger, args, start_exp, checkpoint)


def _export_flipped_scores_summary(logger, args, start_exp, checkpoint):
    scores, flipped_indices, permuted_indices, flipped_train_dataset = (
        _perform_flip_experiment(logger, args, start_exp, checkpoint)
    )
    average_score, std_score = _calculate_average_score(scores, logger=logger)

    # index, label, average_score to csv
    df = pd.DataFrame(average_score.detach().numpy())
    df["index"] = flipped_train_dataset.indices
    df["label"] = flipped_train_dataset.dataset.targets
    df.to_csv(f"{args.save_path}/average_score_{args.timestamp}.csv", index=False)

    # find num_flip samples with the lowest average_score, see how many of them are flipped
    count_flipped_in_lowest_scores(logger, args, flipped_indices, average_score)


def count_flipped_in_lowest_scores(logger, args, flipped_indices, average_score):
    num_flipped_in_lowest_scores = sum(
        idx in flipped_indices for idx in average_score.argsort()[: args.num_flip]
    )
    logger.info(
        f"Number of flipped samples in the lowest {args.num_flip} scores: {num_flipped_in_lowest_scores}"
    )

    return num_flipped_in_lowest_scores


def _calculate_average_score(scores, logger, **kwargs):
    """Calculate the average score for each sample over all experiments."""
    average_score = torch.mean(torch.stack(scores), dim=0)
    std_score = torch.std(torch.stack(scores), dim=0)

    logger.info(f"size of average_score: {average_score.size()}")
    logger.info(f"size of std_score: {std_score.size()}")

    return average_score, std_score


def _perform_flip_experiment(logger, args, start_exp, checkpoint):
    logger.info(
        f"Running flip experiment with {args.num_flip} flips on {args.num_scores} samples"
    )

    scores = []

    for exp in range(start_exp, args.num_exp):
        checkpoint_name = (
            setup_checkpoint_name(args, exp) if args.save_path != "" else ""
        )
        print_experiment_info(args, exp, checkpoint_name)

        (
            flipped_train_dataset,
            test_loader,
            flipped_indices,
            permuted_indices,
            flipped_selection_from,
        ) = initialize_flip_exp(args, args.seed + exp)

        method_class = SELECTION_METHODS.get(args.selection)

        method = method_class(
            flipped_train_dataset,
            args,
            args.fraction,
            args.seed,
            dst_test=test_loader.dataset,
            epochs=args.selection_epochs,
        )

        score = method.get_scores()

        # save the score to file
        # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        try:
            df = pd.DataFrame(score.detach().numpy())
        except AttributeError:
            df = pd.DataFrame(score)
            # for compatibility convert score to tensor
            score = torch.tensor(score)

        df.to_csv(f"{args.save_path}/flip_scores_{exp}.csv", index=False)

        scores.append(score)
        logger.debug(f"Scores: {scores}")

    return scores, flipped_indices, permuted_indices, flipped_train_dataset


def setup_checkpoint_name(args, exp):
    """Set up checkpoint name based on experiment details."""
    return "{dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_{fr}_".format(
        dst=args.dataset,
        net=args.model,
        mtd=args.selection,
        dat=datetime.now(),
        exp=exp,
        epc=args.epochs,
        fr=args.fraction,
    )


def main():
    """
    Main function for running the deep learning experiment, now supporting OTI method.

    This function sets up the experiment, including parsing arguments, loading datasets,
    initializing models, and running the training and evaluation loops. It now includes
    support for the OTI (Online Training Influence) method.

    Key features:
    - Supports both original functionality and new OTI method
    - When using OTI, sets batch size to 1 and saves model parameters after updating each point
    - Calculates scores for each point based on historical parameters and best parameters for OTI

    Returns:
        None
    """

    args = parse_args()
    # 使用命令行参数设置日志级别
    logger = setup_logging(log_level=args.log_level)
    logger.info(f"Parsed arguments: {args}")

    checkpoint, start_exp, start_epoch = setup_experiment(args)
    logger.info(
        f"Experiment setup complete. Starting from experiment {start_exp}, epoch {start_epoch}"
    )

    run_experiment(args, checkpoint, start_exp, start_epoch)

    logger.info("Main function completed")


if __name__ == "__main__":
    # set spawn method for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as e:
        logging.exception("Caught an exception in main:")
        raise
