###
# File: ./utils.py
# Created Date: Saturday, August 24th 2024
# Author: Zihan
# -----
# Last Modified: Friday, 9th May 2025 9:18:25 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np
import pandas as pd
import time, torch
from typing import Dict
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
from datetime import datetime
import os
import sys
import logging
import traceback
import psutil
import json
from datetime import datetime


class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[list(idx)]
        return self.dataset[self.indices[idx]], self.weights[idx]


def train(
    train_loader,
    network,
    criterion,
    optimizer,
    scheduler,
    epoch,
    args,
    rec,
    if_weighted: bool = False,
):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            targets = contents[0][1].to(args.device)
            inputs = contents[0][0].to(args.device)

            # Compute output
            output = network(inputs)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, targets) * weights) / torch.sum(weights)
        else:
            targets = contents[1].to(args.device)
            inputs = contents[0].to(args.device)

            # Compute output
            output = network(inputs)
            loss = criterion(output, targets).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, targets, topk=(1,))[0]
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                )
            )

    record_train_stats(
        rec,
        epoch,
        losses.avg,
        top1.avg,
        optimizer.state_dict()["param_groups"][0]["lr"],
    )


def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, j in enumerate(test_loader):
        (inputs, targets) = j[0], j[1]
        targets = targets.to(args.device)
        inputs = inputs.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(inputs)

            loss = criterion(output, targets).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, targets, topk=(1,))[0]
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(test_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace

    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec


def record_test_stats(rec, step, loss, acc):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    return rec


def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_logging(log_dir="logs", log_level=logging.INFO, log_name=None):
    """
    Setup logging configuration with exception handling

    Args:
        log_dir (str): Directory to store log files
        log_level: Logging level

    Returns:
        logger: Configured logger instance
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成唯一的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_name:
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # 文件处理器
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            # 控制台处理器
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 设置全局异常处理器
    sys.excepthook = handle_exception

    # 创建并配置logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed. Log file: {log_file}")

    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log unhandled exceptions

    Args:
        exc_type: Type of the exception
        exc_value: Exception instance
        exc_traceback: Traceback object
    """
    # 忽略 KeyboardInterrupt 导致的异常
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 获取logger
    logger = logging.getLogger("ExceptionLogger")

    # 创建格式化的错误消息
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # 记录异常
    logger.critical(f"Uncaught exception:\n{error_msg}")


class MemoryMonitor:
    """
    A utility class to monitor and log CPU and GPU memory usage.
    """

    def __init__(self, save_path, method_name):
        self.save_path = save_path
        self.method_name = method_name
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.logger = logging.getLogger(__name__)

        # Create memory logs directory if it doesn't exist
        self.logs_dir = os.path.join(save_path, "memory_logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize log file
        self.log_file = os.path.join(
            self.logs_dir,
            f'memory_usage_{method_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

    def check_memory(self):
        """
        Check current CPU and GPU memory usage and update peaks if necessary.
        Returns:
            tuple: Current (CPU memory in MB, GPU memory in MB)
        """
        # Get CPU memory usage
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_memory)

        # Get GPU memory usage if available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (
                1024 * 1024
            )  # Convert to MB
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory)
            # Reset peak stats for next check
            torch.cuda.reset_peak_memory_stats()

        return cpu_memory, gpu_memory

    def log_memory_usage(self, step=None, extra_info=None):
        """
        Log current memory usage to file with optional step information.

        Args:
            step (str, optional): Current step or phase of the process
            extra_info (dict, optional): Additional information to log
        """
        cpu_memory, gpu_memory = self.check_memory()

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "method": self.method_name,
            "current_cpu_memory_mb": round(cpu_memory, 2),
            "peak_cpu_memory_mb": round(self.peak_cpu_memory, 2),
            "current_gpu_memory_mb": round(gpu_memory, 2),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory, 2),
        }

        if step:
            log_data["step"] = step

        if extra_info:
            log_data.update(extra_info)

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # Log to console
        self.logger.info(
            f"Memory Usage - CPU: {cpu_memory:.2f}MB (Peak: {self.peak_cpu_memory:.2f}MB), "
            f"GPU: {gpu_memory:.2f}MB (Peak: {self.peak_gpu_memory:.2f}MB)"
        )

    def get_summary(self):
        """
        Get a summary of peak memory usage.

        Returns:
            dict: Summary of peak memory usage
        """
        return {
            "method": self.method_name,
            "peak_cpu_memory_mb": round(self.peak_cpu_memory, 2),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory, 2),
        }

    def save_summary(self):
        """
        Save peak memory usage summary to a separate file.
        """
        summary = self.get_summary()
        summary_file = os.path.join(
            self.logs_dir, f"memory_summary_{self.method_name}.json"
        )

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Memory usage summary saved to {summary_file}")


class ScoreTracker:
    """
    A class to track scores across multiple experiments with different seeds.
    """

    def __init__(self, n_samples: int, save_path: str):
        """
        Initialize the score tracker.

        Args:
            n_samples: Number of samples in the dataset
            save_path: Directory to save the results
        """
        self.n_samples = n_samples
        self.save_path = save_path
        self.scores_by_seed: Dict[int, np.ndarray] = {}
        self.current_exp = 0

    def add_scores(self, scores: np.ndarray, seed: int):
        """
        Add scores for a specific seed.

        Args:
            scores: Array of scores for each sample
            seed: Random seed used for the experiment
        """
        self.scores_by_seed[seed] = scores
        self.current_exp += 1

    def compute_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute statistics across all experiments.

        Returns:
            Dictionary containing mean and variance of scores
        """
        all_scores = np.stack(list(self.scores_by_seed.values()))

        return {
            "mean": np.mean(all_scores, axis=0),
            "variance": np.var(all_scores, axis=0),
            "std": np.std(all_scores, axis=0),
        }

    def save_results(self):
        """Save all results to CSV files."""
        # Save individual experiment scores
        for seed, scores in self.scores_by_seed.items():
            df = pd.DataFrame(
                {"index": np.arange(self.n_samples), "score": scores, "seed": seed}
            )
            df.to_csv(
                os.path.join(self.save_path, f"scores_seed_{seed}.csv"), index=False
            )

        # Save statistics
        stats = self.compute_statistics()
        df_stats = pd.DataFrame(
            {
                "index": np.arange(self.n_samples),
                "mean_score": stats["mean"],
                "score_variance": stats["variance"],
                "score_std": stats["std"],
            }
        )
        df_stats.to_csv(
            os.path.join(self.save_path, "score_statistics.csv"), index=False
        )

        # Save combined results
        all_scores = []
        for seed, scores in self.scores_by_seed.items():
            df = pd.DataFrame(
                {"index": np.arange(self.n_samples), "score": scores, "seed": seed}
            )
            all_scores.append(df)

        df_combined = pd.concat(all_scores, ignore_index=True)
        df_combined.to_csv(os.path.join(self.save_path, "all_scores.csv"), index=False)

    def plot_score_distributions(self) -> Dict[str, torch.Tensor]:
        """
        Create visualizations of score distributions and save them.

        Returns:
            Dictionary containing plot data
        """
        stats = self.compute_statistics()
        indices = np.arange(self.n_samples)

        # Sort by mean score
        sort_idx = np.argsort(stats["mean"])[::-1]

        return {
            "indices": indices[sort_idx],
            "mean_scores": stats["mean"][sort_idx],
            "std_scores": stats["std"][sort_idx],
            "variance_scores": stats["variance"][sort_idx],
        }


def custom_collate(batch):
    # 使用 clone().detach() 替代 torch.tensor()
    try:
        data = torch.stack([item[0].clone().detach() for item in batch])
        labels = torch.stack([item[1].clone().detach() for item in batch])
        indices = [item[2] for item in batch]
        return data, labels, indices
    # maybe item[1] is an integer
    except AttributeError:
        data = torch.stack([item[0].clone().detach() for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        indices = [item[2] for item in batch]
        return data, labels, indices

def count_flipped_in_lowest_scores(logger, args, flipped_indices, average_score):
    num_flipped_in_lowest_scores = sum(
        idx in flipped_indices for idx in average_score.argsort()[: args.num_flip]
    )
    logger.info(
        f"Number of flipped samples in the lowest {args.num_flip} scores: {num_flipped_in_lowest_scores}"
    )

    return num_flipped_in_lowest_scores
