###
# File: ./main.py
# Created Date: Monday, October 21st 2024
# Author: Zihan
# -----
# Last Modified: Friday, 9th May 2025 10:35:17 am
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
from liveval.utils import *
from datetime import datetime
from time import sleep
from typing import Type
import logging
import pandas as pd
from experiment_utils import (
    setup_experiment,
    load_checkpoint,
    initialize_dataset_and_model,
    initialize_logging_and_datasets,
    initialize_dataset_properties,
    initialize_flip_exp,
    initialize_network,
    train_and_evaluate_model
)
from logging_utils import setup_logging, get_logger
import sys
from exception_utils import ExceptionHandler, log_exception

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


@log_exception()
def print_experiment_info(args, exp, checkpoint_name):
    """Print the experiment information."""
    logger = logging.getLogger(__name__)

    logger.info(f"================== Exp {exp} ==================")
    logger.info(
        f"exp:{args.exp}, dataset: {args.dataset}, model: {args.model}, selection: {args.selection}, num_ex: {args.num_exp}, epochs: {args.epochs}, fraction: {args.fraction}, seed: {args.seed}, lr: {args.lr}, save_path: {args.save_path}, resume: {args.resume}, device: {args.device}, {'checkpoint_name: {checkpoint_name}' if args.save_path != '' else ''}"
    )


@log_exception()
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


@log_exception()
def _export_flipped_scores_summary(logger, args, start_exp, checkpoint):
    scores, flipped_indices, permuted_indices, flipped_train_dataset = (
        _perform_flip_experiment(logger, args, start_exp, checkpoint)
    )
    average_score, std_score = _calculate_average_score(scores, logger=logger)

    # index, label, average_score to csv
    df = pd.DataFrame(average_score.detach().numpy())
    df["index"] = flipped_train_dataset.indices

    # 获取标签的正确方式
    labels = []
    for idx in flipped_train_dataset.indices:
        _, label, _ = flipped_train_dataset.dataset[idx]
        labels.append(label)
    df["label"] = labels

    df.to_csv(f"{args.save_path}/average_score_{args.timestamp}.csv", index=False)

    # find num_flip samples with the lowest average_score, see how many of them are flipped
    count_flipped_in_lowest_scores(logger, args, flipped_indices, average_score)


@log_exception()
def _calculate_average_score(scores, logger, **kwargs):
    """Calculate the average score for each sample over all experiments."""
    average_score = torch.mean(torch.stack(scores), dim=0)
    std_score = torch.std(torch.stack(scores), dim=0)

    logger.info(f"size of average_score: {average_score.size()}")
    logger.info(f"size of std_score: {std_score.size()}")

    return average_score, std_score


@log_exception()
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


@log_exception()
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


@log_exception()
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
    import torch
    args = parse_args()
    # 使用统一日志接口
    logger = setup_logging(log_level=args.log_level)
    logger.info(f"Parsed arguments: {args}")

    # 记录物理编号用于日志
    physical_gpu_ids = args.gpu.copy() if args.gpu is not None else None
    if args.gpu is not None:
        visible = ",".join(str(i) for i in args.gpu)
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            os.execv(sys.executable, [sys.executable] + sys.argv)
        # 进程重启后，PyTorch只看到指定卡，args.gpu要重映射为逻辑编号
        args.gpu = list(range(torch.cuda.device_count()))

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
        ExceptionHandler().handle(e, context="main")
        raise
