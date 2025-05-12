from experiment.experiment_utils import initialize_flip_exp, find_found_flipped_indices
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
    methods = ["GraNd", "influence_function", "OTI"]
    # 要测试的epoch列表，例如[1, 3, 5, 10, 15]
    epochs_to_test = [1, 2, 3, 4, 5]
    
    # 存储结果的数据结构
    results = {
        "method": [],
        "epochs": [],
        "detection_rate": [],
        "accuracy_before": [],
        "accuracy_after": []
    }
    
    for exp in range(start_exp, args.num_exp):
        # 创建带有翻转标签的数据集
        (
            flipped_train_dataset,
            test_loader,
            flipped_indices,
            permuted_indices,
            flipped_selection_from,
        ) = initialize_flip_exp(args, args.seed + exp)
        
        # 记录原始翻转样本
        pd.DataFrame(flipped_selection_from).to_csv(
            f"{args.save_path}/flipped_selection_from_{args.timestamp}_{exp}.csv",
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
                    flipped_train_dataset,
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
                
                # 确保flipped_indices都在flipped_train_dataset.indices中
                assert all(
                    idx in flipped_train_dataset.indices for idx in flipped_indices
                ), "flipped_indices必须全部在flipped_train_dataset.indices中"
                
                # 统计被找到的翻转点
                found_flipped_indices = find_found_flipped_indices(score, flipped_indices)
                detection_rate = len(found_flipped_indices) / len(flipped_indices)
                
                pd.DataFrame({"found_flipped_indices": found_flipped_indices}).to_csv(
                    f"{args.save_path}/{method_name}_found_flipped_indices_epochs{n_epochs}_{exp}.csv",
                    index=False,
                )
                
                # 构建去掉这些点的新训练集
                remaining_indices = [
                    i for i in flipped_train_dataset.indices if i not in found_flipped_indices
                ]
                new_train_dataset = Subset(flipped_train_dataset.dataset, remaining_indices)
                
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
                    f"Method: {method_name}, Epochs: {n_epochs}, "
                    f"Detection Rate: {detection_rate:.4f}, "
                    f"Accuracy Before: {accuracy_before:.4f}, "
                    f"Accuracy After: {test_acc:.4f}"
                )
    
    # 保存总结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.save_path}/early_detection_results_{args.timestamp}.csv", index=False)
    
    # 生成可视化图表
    create_comparison_visualizations(results_df, args.save_path, args.timestamp)


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


def create_comparison_visualizations(results_df, save_path, timestamp):
    """创建比较不同方法的可视化图表"""
    # 按方法分组
    methods = results_df["method"].unique()

    # 1. 检测率与epoch数的关系
    plt.figure(figsize=(12, 8))
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        plt.plot(method_data["epochs"], method_data["detection_rate"], marker='o', label=method)

    plt.title("Flipped Sample Detection Rate vs Training Epochs")
    plt.xlabel("Number of Training Epochs")
    plt.ylabel("Detection Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/detection_rate_vs_epochs_{timestamp}.png")

    # 2. 准确率提升与epoch数的关系
    plt.figure(figsize=(12, 8))
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        accuracy_improvement = method_data["accuracy_after"] - method_data["accuracy_before"]
        plt.plot(method_data["epochs"], accuracy_improvement, marker='o', label=method)

    plt.title("Accuracy Improvement after Removing Detected Samples vs Training Epochs")
    plt.xlabel("Number of Training Epochs")
    plt.ylabel("Accuracy Improvement")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/accuracy_improvement_vs_epochs_{timestamp}.png")

    # 3. 检测率与准确率提升的关系
    plt.figure(figsize=(12, 8))
    for method in methods:
        method_data = results_df[results_df["method"] == method]
        accuracy_improvement = method_data["accuracy_after"] - method_data["accuracy_before"]
        plt.scatter(method_data["detection_rate"], accuracy_improvement, label=method, s=100)

    plt.title("Accuracy Improvement vs Detection Rate")
    plt.xlabel("Detection Rate")
    plt.ylabel("Accuracy Improvement")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/accuracy_vs_detection_{timestamp}.png")
