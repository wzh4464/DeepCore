###
# File: ./experiment/boundary_detection.py
# Created Date: Tuesday, May 13th 2025
# Author: Zihan
# -----
# Last Modified: Tuesday, 13th May 2025 10:15:23 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date          By      Comments
# ----------    ------  ---------------------------------------------------------
###

from experiment.experiment_utils import initialize_logging_and_datasets
from liveval.utils.utils import count_points_in_lowest_scores
from experiment.experiment_utils import find_found_indices
import pandas as pd
import torch
import numpy as np
import os
from liveval.datasets.boundary_dataset import BoundaryMNISTDataset

def run(args, checkpoint, start_exp, start_epoch):
    """
    使用OTI方法检测Morpho-MNIST边界点
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"运行边界点检测实验，在 {args.num_scores} 个样本中生成 {args.num_boundary} 个边界点"
    )
    scores = []
    for exp in range(start_exp, args.num_exp):
        boundary_dataset, test_loader, boundary_indices, permuted_indices = initialize_boundary_exp(
            args, args.seed + exp
        )
        pd.DataFrame({"boundary_indices": boundary_indices}).to_csv(
            f"{args.save_path}/boundary_indices_{args.timestamp}_{exp}.csv",
            index=False,
        )
        from liveval.methods.selection_methods import SELECTION_METHODS
        method_class = SELECTION_METHODS.get(args.selection)
        method = method_class(
            boundary_dataset,
            args,
            args.fraction,
            args.seed,
            dst_test=test_loader.dataset,
            epochs=args.selection_epochs,
        )
        score = method.get_scores()
        try:
            df = pd.DataFrame(score.detach().numpy())
        except AttributeError:
            df = pd.DataFrame(score)
            score = torch.tensor(score)
        df.to_csv(f"{args.save_path}/boundary_scores_{exp}.csv", index=False)
        scores.append(score)
        assert all(idx in boundary_dataset.indices for idx in boundary_indices), \
            "boundary_indices必须全部在boundary_dataset.indices中"
        found_boundary_indices = find_found_indices(score, boundary_indices)
        pd.DataFrame({"found_boundary_indices": found_boundary_indices}).to_csv(
            f"{args.save_path}/found_boundary_indices_{args.timestamp}_{exp}.csv",
            index=False,
        )
        from torch.utils.data import Subset
        remaining_indices = [
            i for i in boundary_dataset.indices if i not in found_boundary_indices
        ]
        new_train_dataset = Subset(boundary_dataset.dataset, remaining_indices)
        retrain_method = method_class(
            new_train_dataset,
            args,
            args.fraction,
            args.seed,
            dst_test=test_loader.dataset,
            epochs=args.selection_epochs,
        )
        step_losses = []
        epoch_accuracies = []
        def after_loss_hook(outputs, loss, targets, batch_inds, epoch):
            step_losses.append(loss.item())
        def after_epoch_hook():
            if hasattr(retrain_method, "test"):
                acc = getattr(retrain_method, "last_test_acc", None)
                if acc is not None:
                    epoch_accuracies.append(acc)
        retrain_method.after_loss = after_loss_hook
        retrain_method.after_epoch = after_epoch_hook
        test_acc = retrain_method.train_for_epochs(
            args.selection_epochs, remaining_indices, test_loader
        )
        logger.info(f"测试准确率: {test_acc:.4f}")
        pd.DataFrame({"step_loss": step_losses}).to_csv(
            f"{args.save_path}/step_losses_{args.timestamp}_{exp}.csv", index=False
        )
        pd.DataFrame({"epoch_accuracy": epoch_accuracies}).to_csv(
            f"{args.save_path}/epoch_accuracies_{args.timestamp}_{exp}.csv", index=False
        )
    average_score = torch.mean(torch.stack(scores), dim=0)
    df = pd.DataFrame(average_score.detach().numpy())
    df["index"] = boundary_dataset.indices
    labels = []
    for idx in boundary_dataset.indices:
        _, label, _ = boundary_dataset.dataset[idx]
        labels.append(label)
    df["label"] = labels
    df.to_csv(f"{args.save_path}/average_boundary_score_{args.timestamp}.csv", index=False)
    count_points_in_lowest_scores(logger, args, boundary_indices, average_score)

def initialize_boundary_exp(args, seed):
    logger, mean, std, dst_train, dst_test = initialize_logging_and_datasets(args)
    permuted_indices_path = os.path.join(args.save_path, "permuted_indices.csv")
    if os.path.exists(permuted_indices_path):
        permuted_indices = np.loadtxt(permuted_indices_path, delimiter=",", dtype=int)
        logger.info(f"从 {permuted_indices_path} 加载打乱索引")
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_indices = (
            torch.randperm(len(dst_train), generator=generator).numpy().astype(int)
        )
        np.savetxt(permuted_indices_path, permuted_indices, delimiter=",", fmt="%d")
        logger.info(f"保存打乱索引到 {permuted_indices_path}")
    boundary_dataset = BoundaryMNISTDataset(
        dst_train,
        permuted_indices,
        args.num_scores,
        args.num_boundary,
        args.seed,
        logger,
    )
    boundary_indices = boundary_dataset.get_boundary_indices()
    boundary_indices_path = os.path.join(args.save_path, "boundary_indices.csv")
    np.savetxt(boundary_indices_path, boundary_indices, delimiter=",", fmt="%d")
    logger.info(f"边界点索引已保存到 {boundary_indices_path}")
    test_loader = torch.utils.data.DataLoader(
        dst_test,
        batch_size=args.selection_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return boundary_dataset, test_loader, boundary_indices, permuted_indices 