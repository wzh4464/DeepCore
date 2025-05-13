import glob
import warnings
from experiment.experiment_utils import initialize_logging_and_datasets, initialize_boundary_exp, find_found_indices, create_comparison_visualizations
from liveval.utils.utils import count_points_in_lowest_scores
from liveval.methods.selection_methods import SELECTION_METHODS
import pandas as pd
import torch
import numpy as np
import logging
import os
from torch.utils.data import Subset
import matplotlib.pyplot as plt

def run(args, checkpoint, start_exp, start_epoch):
    """
    比较不同方法在检测边界点的能力及移除后的准确率变化
    参数:
        args: 命令行参数
        checkpoint: 检查点信息
        start_exp: 开始实验编号
        start_epoch: 开始的epoch
    """
    logger = logging.getLogger(__name__)
    logger.info(f"运行边界点检测实验，在 {args.num_scores} 个样本中生成 {args.num_boundary} 个边界点")

    # 要比较的方法列表
    methods = ["OTI", "GraNd", "influence_function"]  # 可根据需要修改
    # 要测试的epoch列表
    epochs_to_test = [1, 2, 3, 4, 5]  # 可根据需要修改

    # 存储结果的数据结构
    results = {
        "method": [],
        "epochs": [],
        "detection_rate": [],
        "accuracy_before": [],
        "accuracy_after": []
    }

    for exp in range(start_exp, args.num_exp):
        # 初始化边界数据集和测试加载器
        (
            boundary_dataset,
            test_loader,
            boundary_indices,
            permuted_indices,
            boundary_selection_from,
        ) = initialize_boundary_exp(args, args.seed + exp)

        # 保存边界点索引
        pd.DataFrame({"boundary_indices": boundary_indices}).to_csv(
            f"{args.save_path}/boundary_indices_{exp}.csv",
            index=False,
        )

        # 记录原始边界样本
        pd.DataFrame(boundary_selection_from).to_csv(
            f"{args.save_path}/boundary_selection_from.csv",
            index=False,
        )

        # 对每个方法进行测试
        for method_name in methods:
            # 获取对应的方法类
            method_class = SELECTION_METHODS.get(method_name)

            # 对每个epoch数量进行测试
            for n_epochs in epochs_to_test:
                # 设置当前方法的训练epoch数量
                current_args = args
                current_args.selection_epochs = n_epochs

                # 初始化方法
                method = method_class(
                    boundary_dataset,
                    current_args,
                    current_args.fraction,
                    current_args.seed,
                    dst_test=test_loader.dataset,
                    epochs=n_epochs,
                )

                # 获取分数并保存
                score = method.get_scores()
                try:
                    df = pd.DataFrame(score.detach().numpy())
                except AttributeError:
                    df = pd.DataFrame(score)
                    score = torch.tensor(score)

                df.to_csv(f"{args.save_path}/{method_name}_scores_epochs{n_epochs}_{exp}.csv", index=False)

                # 确保boundary_indices都在boundary_dataset.indices中
                assert all(
                    idx in boundary_dataset.indices for idx in boundary_indices
                ), "boundary_indices必须全部在boundary_dataset.indices中"

                # 找出被检测到的边界点
                found_boundary_indices = find_found_indices(score, boundary_indices)
                detection_rate = len(found_boundary_indices) / len(boundary_indices)

                pd.DataFrame({"found_boundary_indices": found_boundary_indices}).to_csv(
                    f"{args.save_path}/{method_name}_found_boundary_indices_epochs{n_epochs}_{exp}.csv",
                    index=False,
                )

                # 构建去掉这些点的新训练集
                remaining_indices = [
                    i for i in boundary_dataset.indices if i not in found_boundary_indices
                ]
                new_train_dataset = Subset(boundary_dataset.dataset, remaining_indices)

                # 记录移除前的测试准确率
                accuracy_before = evaluate_model(method, test_loader)

                # 用新数据集重新训练
                retrain_method = method_class(
                    new_train_dataset,
                    current_args,
                    current_args.fraction,
                    current_args.seed,
                    dst_test=test_loader.dataset,
                    epochs=current_args.selection_epochs,
                )

                # 记录训练过程中的loss和accuracy
                step_losses = []
                epoch_accuracies = []

                # 设置hook记录loss/accuracy
                def after_loss_hook(outputs, loss, targets, batch_inds, epoch):
                    step_losses.append(loss.item())

                def after_epoch_hook():
                    if hasattr(retrain_method, "test"):
                        acc = getattr(retrain_method, "last_test_acc", None)
                        if acc is not None:
                            epoch_accuracies.append(acc)

                retrain_method.after_loss = after_loss_hook
                retrain_method.after_epoch = after_epoch_hook

                # 训练模型
                test_acc = retrain_method.train_for_epochs(
                    current_args.selection_epochs, remaining_indices, test_loader
                )

                # 记录结果
                results["method"].append(method_name)
                results["epochs"].append(n_epochs)
                results["detection_rate"].append(detection_rate)
                results["accuracy_before"].append(accuracy_before)
                results["accuracy_after"].append(test_acc)

                # 保存loss/accuracy
                pd.DataFrame({"step_loss": step_losses}).to_csv(
                    f"{args.save_path}/{method_name}_step_losses_epochs{n_epochs}_{exp}.csv", index=False
                )
                pd.DataFrame({"epoch_accuracy": epoch_accuracies}).to_csv(
                    f"{args.save_path}/{method_name}_epoch_accuracies_epochs{n_epochs}_{exp}.csv", index=False
                )

                logger.info(
                    f"方法: {method_name}, Epochs: {n_epochs}, "
                    f"检测率: {detection_rate:.4f}, "
                    f"移除前准确率: {accuracy_before:.4f}, "
                    f"移除后准确率: {test_acc:.4f}"
                )

        # 计算并记录当前实验的平均结果
        exp_results = pd.DataFrame(results)
        exp_avg = exp_results.groupby(['method', 'epochs']).mean().reset_index()
        logger.info(f"实验 {exp} 的平均结果:")
        for _, row in exp_avg.iterrows():
            logger.info(
                f"方法: {row['method']}, Epochs: {int(row['epochs'])}, "
                f"平均检测率: {row['detection_rate']:.4f}, "
                f"平均移除前准确率: {row['accuracy_before']:.4f}, "
                f"平均移除后准确率: {row['accuracy_after']:.4f}"
            )

    # 保存总结果
    results_df = pd.DataFrame(results)
    results_dirname = f"{args.save_path}/boundary_detection_results.csv"
    results_df.to_csv(results_dirname, index=False)

    # 生成可视化图表
    create_comparison_visualizations(results_dirname, args.save_path)
    
    # 对所有方法和epoch的平均分数进行额外分析
    for method_name in methods:
        method_results = results_df[results_df["method"] == method_name]
        average_detection_rate = method_results["detection_rate"].mean()
        average_accuracy_improvement = (method_results["accuracy_after"] - method_results["accuracy_before"]).mean()
        logger.info(f"方法 {method_name} 的总体表现:")
        logger.info(f"平均检测率: {average_detection_rate:.4f}")
        logger.info(f"平均准确率提升: {average_accuracy_improvement:.4f}")


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
    logger.info(f"运行检测率与epoch关系图绘制")

    # 读取结果数据
    # 自动查找最新的 boundary_detection_results.csv
    result_files = glob.glob(f"{args.save_path}/boundary_detection_results.csv")
    if not result_files:
        raise FileNotFoundError(f"未找到 {args.save_path}/boundary_detection_results.csv 文件")
    # 提取所有 timestamp
    timestamps = []
    for f in result_files:
        base = os.path.basename(f)
        try:
            ts = base.split("boundary_detection_results_")[1].split(".csv")[0]
            timestamps.append(ts)
        except Exception:
            continue
    if not timestamps:
        raise ValueError("未能从文件名中提取到 timestamp")
    if len(timestamps) > 1:
        warnings.warn(f"检测到多个 timestamp: {timestamps}，将使用最新的 {max(timestamps)}")
    timestamp = max(timestamps)
    results_dirname = f"{args.save_path}/boundary_detection_results.csv"

    create_comparison_visualizations(results_dirname, args.save_path)
    
    # 添加额外的边界点特定分析
    results_df = pd.read_csv(results_dirname)
    for method in results_df["method"].unique():
        method_df = results_df[results_df["method"] == method]
        plt.figure(figsize=(10, 6))
        plt.plot(method_df["epochs"], method_df["detection_rate"], 'o-', label="检测率")
        plt.plot(method_df["epochs"], method_df["accuracy_after"], 's-', label="移除后准确率")
        plt.title(f"方法 {method} 的性能随训练epoch变化")
        plt.xlabel("训练Epochs")
        plt.ylabel("指标值")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{args.save_path}/{method}_performance_vs_epochs_{timestamp}.png")
        plt.close()


def evaluate_model(method, test_loader):
    """评估模型在测试集上的准确率"""
    if not hasattr(method, "model"):
        return 0.0

    method.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(method.args.device), targets.to(method.args.device)
            outputs = method.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total if total > 0 else 0.0
