###
# File: ./experiment_utils.py
# Created Date: Friday, May 9th 2025
# Author: Zihan
# -----
# Last Modified: Tuesday, 13th May 2025 9:52:04 am
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
from liveval.utils.utils import load_checkpoint, save_best_checkpoint, finalize_checkpoint
from liveval.utils.exception_utils import log_exception, ExceptionHandler
import matplotlib.pyplot as plt
import glob
import warnings

# 迁移自 main.py

plt.rcParams["font.family"] = "Times New Roman"

@log_exception()
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


@log_exception()
def initialize_logging_and_datasets(args):
    logger = logging.getLogger(__name__)
    mean, std, dst_train, dst_test = initialize_dataset_properties(args)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    return logger, mean, std, dst_train, dst_test


@log_exception()
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


@log_exception()
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
    if args.exp in ["flip", "early_detection"]:
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


@log_exception()
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


@log_exception()
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
    from experiment.experiment_utils import initialize_network
    from liveval.utils.utils import (
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

def find_found_flipped_indices(score, flipped_indices, num_to_select=None):
    """
    根据分数和反转点索引，返回被找到的反转点索引列表。
    score: tensor 或 numpy array，所有样本的分数。
    flipped_indices: list 或 array，反转点的索引。
    num_to_select: 选取最低分数的样本数，默认与flipped_indices数量一致。
    返回：被找到的反转点索引列表。
    """
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    if num_to_select is None:
        num_to_select = len(flipped_indices)
    # 取最低分数的样本索引
    lowest_score_indices = np.argpartition(score, num_to_select)[:num_to_select]
    # 找到这些最低分数中哪些是反转点
    found_flipped_indices = [i for i in lowest_score_indices if i in flipped_indices]
    return found_flipped_indices

def find_found_indices(score, target_indices, num_to_select=None):
    """
    根据分数和目标点索引，返回被找到的目标点索引列表。
    score: tensor 或 numpy array，所有样本的分数。
    target_indices: list 或 array，目标点的索引。
    num_to_select: 选取最低分数的样本数，默认与target_indices数量一致。
    返回：被找到的目标点索引列表。
    """
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    if num_to_select is None:
        num_to_select = len(target_indices)
    lowest_score_indices = np.argpartition(score, num_to_select)[:num_to_select]
    found_indices = [i for i in lowest_score_indices if i in target_indices]
    return found_indices

def create_comparison_visualizations(results_dirname, save_path):
    """创建比较不同方法的可视化图表（使用PlotManager），自动从save_path读取最新timestamp"""
    # 自动查找最新的 early_detection_results.csv
    import pandas as pd
    results_df = pd.read_csv(results_dirname, index_col=False)
    result_files = glob.glob(f"{save_path}/early_detection_results.csv")
    if not result_files:
        raise FileNotFoundError(f"未找到 {save_path}/early_detection_results.csv 文件")
    # 提取所有 timestamp
    # timestamps = []
    # for f in result_files:
    #     # 文件名格式 early_detection_results.csv
    #     base = os.path.basename(f)
    #     try:
    #         ts = base.split("early_detection_results_")[1].split(".csv")[0]
    #         timestamps.append(ts)
    #     except Exception:
    #         continue
    # if not timestamps:
    #     raise ValueError("未能从文件名中提取到 timestamp")
    # if len(timestamps) > 1:
    #     warnings.warn(f"检测到多个 timestamp: {timestamps}，将使用最新的 {max(timestamps)}")
    methods = results_df["method"].unique()

    # 1. 检测率与epoch数的关系
    pm1 = PlotManager()
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        pm1.plot(method_data["epochs"], method_data["detection_rate"], label=method)
    set_plot_labels_and_legend(
        pm1,
        "Number of Training Epochs",
        "Detection Rate",
        "Flipped Sample Detection Rate vs Training Epochs",
    )
    # timestamp = max(timestamps)
    pm1.savefig(f"{save_path}/detection_rate_vs_epochs.png")
    pm1.close()

    # 2. 准确率提升与epoch数的关系
    pm2 = PlotManager()
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        accuracy_improvement = method_data["accuracy_after"] - method_data["accuracy_before"]
        pm2.plot(method_data["epochs"], accuracy_improvement, label=method)
    set_plot_labels_and_legend(
        pm2,
        "Number of Training Epochs",
        "Accuracy Improvement",
        "Accuracy Improvement after Removing Detected Samples vs Training Epochs",
    )
    pm2.savefig(f"{save_path}/accuracy_improvement_vs_epochs.png")
    pm2.close()

    # 3. 检测率与准确率提升的关系
    pm3 = PlotManager()
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        accuracy_improvement = method_data["accuracy_after"] - method_data["accuracy_before"]
        pm3.scatter(method_data["detection_rate"], accuracy_improvement, label=method)
    set_plot_labels_and_legend(
        pm3,
        "Detection Rate",
        "Accuracy Improvement",
        "Accuracy Improvement vs Detection Rate",
    )
    pm3.savefig(f"{save_path}/accuracy_vs_detection.png")
    pm3.close()

def set_plot_labels_and_legend(plot_manager, xlabel, ylabel, title):
    plot_manager.set_xlabel(xlabel)
    plot_manager.set_ylabel(ylabel)
    plot_manager.set_title(title)
    plot_manager.add_legend()
    plot_manager.add_grid()

class PlotManager:
    """
    画图管理器，支持灵活设置画图类型、数据、颜色、字体、字号、线宽、legend、label、x/y范围等参数。
    默认参数参考 cleansing_plot.py。
    """
    def __init__(self,
                 fig_size=(12, 8),
                 font_family='Times New Roman',
                 font_size=28,
                 axes_titlesize=28,
                 axes_labelsize=28,
                 xtick_labelsize=26,
                 ytick_labelsize=26,
                 legend_fontsize=26,
                 color_list=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        # 默认颜色
        if color_list is None:
            self.color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
        else:
            self.color_list = color_list
        # 设置全局字体
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.titlesize'] = axes_titlesize
        plt.rcParams['axes.labelsize'] = axes_labelsize
        plt.rcParams['xtick.labelsize'] = xtick_labelsize
        plt.rcParams['ytick.labelsize'] = ytick_labelsize
        plt.rcParams['legend.fontsize'] = legend_fontsize
        self.line_handles = []
        self.label_handles = []
        self.color_idx = 0

    def plot(self, x, y, label=None, color=None, linewidth=4, linestyle='-', marker=None, **kwargs):
        if color is None:
            color = self.color_list[self.color_idx % len(self.color_list)]
            self.color_idx += 1
        line, = self.ax.plot(x, y, linestyle=linestyle, marker=marker, label=label, color=color, linewidth=linewidth, **kwargs)
        if label is not None:
            self.line_handles.append(line)
            self.label_handles.append(label)
        return line

    def scatter(self, x, y, label=None, color=None, s=100, **kwargs):
        if color is None:
            color = self.color_list[self.color_idx % len(self.color_list)]
            self.color_idx += 1
        sc = self.ax.scatter(x, y, label=label, color=color, s=s, **kwargs)
        if label is not None:
            self.line_handles.append(sc)
            self.label_handles.append(label)
        return sc

    def plot_with_std(self, x, y_means, y_stds, label=None, color=None, linewidth=4, 
                      marker='o', alpha=0.2, fill_between=True, **kwargs):
        """
        绘制带有标准差区域的曲线
        
        参数:
            x: x轴数据
            y_means: y轴平均值数据
            y_stds: y轴标准差数据
            label: 图例标签
            color: 线条颜色
            linewidth: 线条宽度
            marker: 数据点标记
            alpha: 标准差区域透明度
            fill_between: 是否填充标准差区域
            **kwargs: 其他plot参数
        
        返回:
            matplotlib Line2D对象
        """
        import numpy as np
        
        if color is None:
            color = self.color_list[self.color_idx % len(self.color_list)]
            self.color_idx += 1
            
        # 绘制主曲线
        line = self.plot(x, y_means, label=label, color=color, 
                         linewidth=linewidth, marker=marker, **kwargs)
        
        # 绘制标准差区域
        if fill_between and y_stds is not None:
            # 确保下界不小于0
            lower_bound = np.maximum(0, np.array(y_means) - np.array(y_stds))
            upper_bound = np.array(y_means) + np.array(y_stds)
            
            self.ax.fill_between(x, lower_bound, upper_bound, 
                                alpha=alpha, color=color)
        
        return line

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)

    def set_title(self, title):
        self.ax.set_title(title)

    def set_xlim(self, xmin=None, xmax=None):
        self.ax.set_xlim(left=xmin, right=xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self.ax.set_ylim(bottom=ymin, top=ymax)

    def add_legend(self, handles=None, labels=None, loc='best', **kwargs):
        if handles is None:
            handles = self.line_handles
        if labels is None:
            labels = self.label_handles
        self.ax.legend(handles=handles, labels=labels, loc=loc, **kwargs)

    def add_grid(self, which='both', axis='both', **kwargs):
        self.ax.grid(which=which, axis=axis, **kwargs)

    def savefig(self, path, dpi=300, tight=True):
        if tight:
            self.plt.tight_layout()
        self.fig.savefig(path, dpi=dpi)

    def clear(self):
        self.ax.cla()
        self.line_handles = []
        self.label_handles = []
        self.color_idx = 0

    def close(self):
        self.plt.close(self.fig)

    def set_xticks_int(self):
        """
        设置x轴刻度为整数。
        """
        import matplotlib.ticker as ticker
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

@log_exception()
def initialize_boundary_exp(args, seed):
    """
    初始化边界点实验
    参数:
        args: 命令行参数
        seed: 随机种子
    返回:
        boundary_dataset: 包含边界点的数据集
        test_loader: 测试数据加载器
        boundary_indices: 边界点索引
        permuted_indices: 打乱的索引
        boundary_selection_from: 用于计算分数的索引
    """
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
    from liveval.datasets.boundary_dataset import BoundaryDataset
    boundary_dataset = BoundaryDataset(
        dst_train,
        permuted_indices,
        args.num_scores,
        args.num_boundary,
        args.dataset,
        args.seed,
        logger,
        transform_intensity=getattr(args, 'boundary_transform_intensity', 0.5),
    )
    boundary_indices = boundary_dataset.get_boundary_indices()
    boundary_selection_from = boundary_dataset.get_boundary_selection_from()
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
    return (
        boundary_dataset,
        test_loader,
        boundary_indices,
        permuted_indices,
        boundary_selection_from,
    )
