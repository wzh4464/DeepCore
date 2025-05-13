import os
import torch
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Sampler
from experiment.experiment_utils import initialize_dataset_and_model
from liveval.methods.selection_methods import SELECTION_METHODS

class OrderedSampler(Sampler):
    """按照给定顺序采样的采样器"""
    def __init__(self, indices):
        self.indices = indices
        
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def generate_different_orders(n_samples, n_orders=5, seed=42):
    """生成多种不同的训练顺序"""
    np.random.seed(seed)
    orders = []
    
    # 顺序1: 原始顺序
    orders.append(np.arange(n_samples))
    
    # 顺序2: 完全随机
    random_order = np.arange(n_samples)
    np.random.shuffle(random_order)
    orders.append(random_order)
    
    # 顺序3-n: 其他随机顺序
    for i in range(n_orders - 2):
        new_order = np.arange(n_samples)
        np.random.seed(seed + i + 1)
        np.random.shuffle(new_order)
        orders.append(new_order)
    
    return orders

def run(args, checkpoint, start_exp, start_epoch):
    """运行TracIn顺序实验"""
    logger = logging.getLogger(__name__)
    logger.info("Running TracIn order experiment")
    
    # 加载数据集
    train_loader, test_loader, if_weighted, subset, selection_args = initialize_dataset_and_model(args, checkpoint)
    
    # 获取数据集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    # 获取样本数量
    n_samples = len(train_dataset)
    
    # 生成不同的训练顺序
    n_orders = args.num_exp  # 使用命令行参数中的num_exp作为顺序数量
    orders = generate_different_orders(n_samples, n_orders, args.seed)
    
    # 存储所有顺序的分数
    all_scores = []
    
    # 对每种顺序运行实验
    for i, order in enumerate(orders):
        logger.info(f"Running experiment with order {i+1}/{n_orders}")
        
        # 创建自定义数据加载器，确保按照指定顺序加载数据
        custom_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            sampler=OrderedSampler(order),
            num_workers=args.workers,
            pin_memory=True
        )
        
        # 初始化TracIn方法
        tracin = SELECTION_METHODS["TracIn"](
            train_dataset,
            args,
            fraction=args.fraction,
            random_seed=args.seed + i,  # 每个实验使用不同的随机种子
            epochs=args.selection_epochs,
            specific_model=args.model,
            checkpoint_interval=1,  # 每个epoch保存一个检查点
            num_test_samples=args.num_scores,  # 使用命令行参数中的num_scores
            dst_test=test_dataset
        )
        
        # 保存训练顺序
        order_df = pd.DataFrame({"index": np.arange(n_samples), "order": order})
        order_df.to_csv(os.path.join(args.save_path, f"order_{i}.csv"), index=False)
        
        # 使用当前顺序训练模型
        tracin.train_loader = custom_train_loader
        tracin.before_run()
        tracin.run()
        
        # 计算TracIn scores
        scores = tracin.get_scores()
        
        # 转换为numpy数组
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # 保存分数
        scores_df = pd.DataFrame({
            "index": np.arange(n_samples),
            "order_position": np.argsort(order),  # 样本在顺序中的位置
            "tracin_score": scores
        })
        scores_df.to_csv(os.path.join(args.save_path, f"tracin_scores_order_{i}.csv"), index=False)
        
        all_scores.append(scores)
    
    # 分析结果
    analyze_tracin_order_results(args, n_orders, all_scores, n_samples)

def analyze_tracin_order_results(args, n_orders, all_scores, n_samples):
    """分析不同顺序对TracIn scores的影响"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing TracIn order experiment results")
    
    # 创建结果数据框
    scores_data = []
    for i in range(n_orders):
        for j in range(n_samples):
            scores_data.append({
                "sample_index": j,
                "order_id": i,
                "tracin_score": all_scores[i][j]
            })
    
    df = pd.DataFrame(scores_data)
    
    # 保存完整结果
    df.to_csv(os.path.join(args.save_path, "all_tracin_scores.csv"), index=False)
    
    # 透视表以样本为行，顺序为列
    pivot_df = df.pivot(index="sample_index", columns="order_id", values="tracin_score")
    
    # 计算每个样本在不同顺序下的统计量
    pivot_df["mean"] = pivot_df.mean(axis=1)
    pivot_df["std"] = pivot_df.std(axis=1)
    pivot_df["cv"] = pivot_df["std"] / pivot_df["mean"].abs()  # 变异系数
    pivot_df["max_diff"] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
    pivot_df["range_ratio"] = pivot_df["max_diff"] / pivot_df["mean"].abs()
    
    # 保存统计结果
    pivot_df.to_csv(os.path.join(args.save_path, "tracin_scores_stats.csv"))
    
    # 创建可视化
    plt.figure(figsize=(16, 12))
    
    # 1. 不同顺序下各样本的分数分布
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x="order_id", y="tracin_score")
    plt.title("Distribution of TracIn Scores Across Different Orders")
    plt.xlabel("Order ID")
    plt.ylabel("TracIn Score")
    
    # 2. 样本分数的标准差分布
    plt.subplot(2, 2, 2)
    sns.histplot(pivot_df["std"], kde=True)
    plt.title("Distribution of Standard Deviation of TracIn Scores")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Count")
    
    # 3. 变异系数分布
    plt.subplot(2, 2, 3)
    sns.histplot(pivot_df["cv"], kde=True)
    plt.title("Distribution of Coefficient of Variation")
    plt.xlabel("Coefficient of Variation")
    plt.ylabel("Count")
    
    # 4. 样本在不同顺序下的分数变化
    plt.subplot(2, 2, 4)
    sample_indices = np.random.choice(n_samples, min(10, n_samples), replace=False)
    for idx in sample_indices:
        plt.plot(range(n_orders), pivot_df.iloc[idx, :n_orders], marker='o', label=f"Sample {idx}")
    plt.title("TracIn Scores of Selected Samples Across Orders")
    plt.xlabel("Order ID")
    plt.ylabel("TracIn Score")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_order_analysis.png"))
    
    # 打印统计摘要
    logger.info("TracIn Order Experiment Results Summary:")
    logger.info(f"平均标准差: {pivot_df['std'].mean()}")
    logger.info(f"最大标准差: {pivot_df['std'].max()}")
    logger.info(f"平均变异系数: {pivot_df['cv'].mean()}")
    logger.info(f"变异系数超过0.5的样本比例: {(pivot_df['cv'] > 0.5).mean()}")
    logger.info(f"最大分数差异与平均值比率超过1的样本比例: {(pivot_df['range_ratio'] > 1).mean()}") 