###
# File: ./experiment_utils.py
# Created Date: Friday, May 9th 2025
# Author: Zihan
# -----
# Last Modified: Friday, 9th May 2025 9:58:06 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import logging
import torch
import numpy as np
from torchvision import transforms
import liveval.datasets as datasets
import liveval.nets as nets
from torch import nn
from liveval.datasets.flipped_dataset import FlippedDataset
from liveval.datasets.corrupted_dataset import CorruptedDataset
from liveval.methods.selection_methods import SELECTION_METHODS
from liveval.utils import load_checkpoint, save_best_checkpoint, finalize_checkpoint

# 迁移自 main.py

def setup_experiment(args):
    """
    Set up directories, checkpoint, and batch sizes.
    """
    logger = logging.getLogger(__name__)
    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        logger.info(f"Created directory: {args.save_path}")
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)
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


def initialize_logging_and_datasets(args):
    logger = logging.getLogger(__name__)
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
    logger, mean, std, dst_train, dst_test = initialize_logging_and_datasets(args)
    permuted_indices_path = os.path.join(args.save_path, "permuted_indices.csv")
    if os.path.exists(permuted_indices_path):
        permuted_indices = np.loadtxt(permuted_indices_path, delimiter=",").astype(int)
        logger.info(f"Loaded permuted indices from {permuted_indices_path}")
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_indices = (
            torch.randperm(len(dst_train), generator=generator).numpy().astype(int)
        )
        np.savetxt(permuted_indices_path, permuted_indices, delimiter=",", fmt="%d")
        logger.info(f"Saved permuted indices to {permuted_indices_path}")
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
    flipped_indices_path = os.path.join(args.save_path, "flipped_indices.csv")
    np.savetxt(flipped_indices_path, flipped_indices, delimiter=",", fmt="%d")
    logger.info(f"Saved flipped indices to {flipped_indices_path}")
    test_loader = torch.utils.data.DataLoader(
        dst_test,
        batch_size=args.selection_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return (
        flipped_dataset,
        test_loader,
        flipped_indices,
        permuted_indices,
        flipped_selection_from,
    )


def initialize_network(args, model, train_loader, checkpoint, start_epoch):
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
            available_methods = list(SELECTION_METHODS.keys())
            raise ValueError(
                f"Selection method {args.selection} not found. 可用的方法有: {available_methods}"
            )

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
    if if_weighted:
        raise NotImplementedError("Weighted subset is not implemented yet")
    else:
        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

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
    from experiment_utils import initialize_network
    from liveval.utils import (
        train,
        test,
        save_checkpoint,
        record_ckpt,
        init_recorder,
    )
    import os
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
