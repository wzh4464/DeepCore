import glob
import warnings
from experiment.experiment_utils import (
    initialize_flip_exp,
    find_found_flipped_indices,
    create_comparison_visualizations,
)
from liveval.methods.selection_methods import SELECTION_METHODS
import pandas as pd
from experiment.experiment_manager import ExperimentManager
import torch
import numpy as np
import logging
import os
from torch.utils.data import Subset
import matplotlib.pyplot as plt


def run(args, checkpoint, start_exp, start_epoch):
    """
    比较不同方法在早期训练阶段检测flipped样本的能力及移除后的准确率变化
    参数:
        args: 命令行参数
        checkpoint: 检查点信息
        start_exp: 开始实验编号
        start_epoch: 开始的epoch
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running early detection comparison with {args.num_flip} flips")

    # 要比较的方法列表
    # methods = ["OTI", "GraNd", "influence_function"]
    method_name = args.selection  # 从args获取当前选择的方法
    methods = [method_name]  # 只测试当前指定的方法
    # 要测试的epoch列表，例如[1, 3, 5, 10, 15]
    epochs_to_test = [1, 2, 3, 4, 5]

    # 存储结果的数据结构
    results = {
        "method": [],
        "epochs": [],
        "detection_rate": [],
        "accuracy_before": [],
        "accuracy_after": [],
    }

    # 初始化实验管理器
    experiment_manager = ExperimentManager(args)

    for exp in range(start_exp, args.num_exp):
        # 使用实验管理器准备数据集
        dataset_result = experiment_manager.prepare_flipped_dataset(args.seed + exp)
        flipped_train_dataset = dataset_result["dataset"]
        test_loader = dataset_result["test_loader"]
        flipped_indices = dataset_result["special_indices"]["flipped"]
        flipped_selection_from = dataset_result["special_indices"]["selection"]

        # 记录原始翻转样本
        pd.DataFrame(flipped_selection_from).to_csv(
            f"{args.save_path}/flipped_selection_from.csv",
            index=False,
        )

        # 对每个方法进行测试
        for method_name in methods:
            # 获取方法
            for n_epochs in epochs_to_test:
                method = experiment_manager.prepare_method(
                    method_name, flipped_train_dataset, test_loader
                )
                result = experiment_manager.evaluate_method(
                    method,
                    n_epochs,
                    flipped_indices,
                    flipped_train_dataset,
                    test_loader,
                )

                # 记录结果
                results["method"].append(method_name)
                results["epochs"].append(n_epochs)
                results["detection_rate"].append(result["detection_rate"])
                results["accuracy_before"].append(result["accuracy_before"])
                results["accuracy_after"].append(result["accuracy_after"])

                logger.info(
                    f"Method: {method_name}, Epochs: {n_epochs}, "
                    f"Detection Rate: {result['detection_rate']:.4f}, "
                    f"Accuracy Before: {result['accuracy_before']:.4f}, "
                    f"Accuracy After: {result['accuracy_after']:.4f}"
                )

    # 保存总结果
    results_df = pd.DataFrame(results)
    results_dirname = f"{args.save_path}/early_detection_results.csv"
    results_df.to_csv(results_dirname, index=False)

    # 生成可视化图表
    create_comparison_visualizations(results_dirname, args.save_path)


def plot_detection_rate_vs_epochs(args, checkpoint, start_exp, start_epoch):
    """
    绘制检测率与训练epoch数的关系图
    参数:
        args: 命令行参数
        checkpoint: 检查点信息
        start_exp: 开始实验编号
        start_epoch: 开始的epoch
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running plot detection rate vs epochs")

    # 读取结果数据
    # results_dirname = f"{args.save_path}/early_detection_results.csv"
    results_dirname_prefix = f"{args.save_path}/early_detection_results_"
    # 自动查找最新的 early_detection_results.csv
    result_files = glob.glob(f"{args.save_path}/early_detection_results.csv")
    if not result_files:
        raise FileNotFoundError(
            f"未找到 {args.save_path}/early_detection_results.csv 文件"
        )
    # 提取所有 timestamp
    # timestamps = []
    # for f in result_files:
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
    # timestamp = max(timestamps)
    results_dirname = f"{args.save_path}/early_detection_results.csv"

    create_comparison_visualizations(results_dirname, args.save_path)


def evaluate_model(method, test_loader):
    """评估模型在测试集上的准确率"""
    if not hasattr(method, "model"):
        return 0.0

    method.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(method.args.device), targets.to(
                method.args.device
            )
            outputs = method.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total if total > 0 else 0.0
