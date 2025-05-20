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

def generate_specific_orders(n_samples, num_selected_samples=100, seed=42):
    """
    生成两种特定的训练顺序:
    1. 选定的 num_selected_samples 个样本在最前面.
    2. 选定的 num_selected_samples 个样本在最后面.
    选定的样本是固定的 (例如，前 num_selected_samples 个，或基于seed随机选择一次后固定).
    """
    logger = logging.getLogger(__name__)
    if num_selected_samples > n_samples:
        raise ValueError("选择的样本数不能大于总样本数")

    # 为了可重复性，如果需要随机选择，请固定种子
    np.random.seed(seed) 
    
    all_indices = np.arange(n_samples)
    
    # 选择固定的100个点 (这里简单选择前100个，也可以根据需要修改为随机选择)
    # selected_indices = np.random.choice(all_indices, num_selected_samples, replace=False)
    if n_samples >= num_selected_samples:
        selected_indices = all_indices[:num_selected_samples]
    else:
        # 如果总样本数不足100，则选择所有样本
        selected_indices = all_indices
        logger.warning(f"总样本数 ({n_samples}) 小于请求的选择样本数 ({num_selected_samples})。将选择所有样本。")
        num_selected_samples = n_samples # 更新实际选择的样本数

    remaining_indices = np.setdiff1d(all_indices, selected_indices, assume_unique=True)
    
    # 保持剩余索引的原始相对顺序
    # np.random.shuffle(remaining_indices) # 如果希望剩余部分也随机打乱，取消此行注释

    order_selected_first = np.concatenate([selected_indices, remaining_indices])
    order_selected_last = np.concatenate([remaining_indices, selected_indices])
    
    orders = [order_selected_first, order_selected_last]
    order_names = ["selected_first", "selected_last"]
    
    # 保存选择的样本索引，便于后续分析
    # selected_samples_df = pd.DataFrame({"selected_indices": selected_indices})
    # save_path = getattr(args, "save_path", ".") # 获取保存路径
    # selected_samples_df.to_csv(os.path.join(save_path, "selected_100_samples_indices.csv"), index=False)
    # logger.info(f"已选择 {num_selected_samples} 个样本，索引已保存。")


    return orders, order_names, selected_indices

def run(args, checkpoint, start_exp, start_epoch):
    """运行TracIn顺序实验"""
    logger = logging.getLogger(__name__)
    logger.info("Running TracIn specific order experiment for stability check")

    # 加载数据集
    train_loader, test_loader, if_weighted, subset, selection_args = initialize_dataset_and_model(args, checkpoint)

    # 获取数据集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # 获取样本数量
    n_samples = len(train_dataset)

    # 定义要选择的样本数量
    num_selected_samples = 100 # 用户要求的100个点

    # 生成特定的训练顺序
    orders, order_names, selected_indices = generate_specific_orders(n_samples, num_selected_samples, args.seed)
    n_orders = len(orders) # 现在是2

    # 将selected_indices 保存到文件
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        selected_samples_df = pd.DataFrame({"selected_indices": selected_indices})
        selected_samples_df.to_csv(os.path.join(args.save_path, "selected_samples_indices.csv"), index=False)
        logger.info(f"Selected {len(selected_indices)} sample indices saved to {os.path.join(args.save_path, 'selected_samples_indices.csv')}")

    # 存储所有顺序的分数
    all_scores_dict = {} # 使用字典存储，键为 order_name

    # 对每种顺序运行实验
    for i, order_name in enumerate(order_names):
        current_order = orders[i]
        logger.info(f"Running experiment with order: {order_name} ({i+1}/{n_orders})")

        # 创建自定义数据加载器，确保按照指定顺序加载数据
        custom_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            sampler=OrderedSampler(current_order),
            num_workers=args.workers,
            pin_memory=True
        )

        # 初始化TracIn方法
        tracin = SELECTION_METHODS["TracIn"](
            train_dataset,
            args,
            fraction=args.fraction,  # fraction可能不再直接相关，因为我们是全量计算分数
            random_seed=args.seed + i,
            epochs=args.selection_epochs,
            specific_model=args.model,
            checkpoint_interval=1,
            num_test_samples=args.num_scores,
            dst_test=test_dataset,
        )

        # 保存训练顺序
        # order_df = pd.DataFrame({"index": np.arange(n_samples), "order_position_in_current_run": current_order }) # 保存的是原始索引在此次运行中的顺序
        # order_df.to_csv(os.path.join(args.save_path, f"order_{order_name}.csv"), index=False)

        # 使用当前顺序训练模型
        tracin.train_loader = custom_train_loader # 确保TracIn使用我们的自定义加载器
        tracin.before_run()
        tracin.run()

        # 计算TracIn scores (获取所有样本的分数)
        scores = tracin.get_scores() 

        # 转换为numpy数组
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        # 保存分数: 原始索引 -> 分数
        # scores_df = pd.DataFrame({
        #     "original_index": np.arange(n_samples),
        #     "tracin_score": scores
        # })
        # scores_df.to_csv(os.path.join(args.save_path, f"tracin_scores_{order_name}.csv"), index=False)

        all_scores_dict[order_name] = scores # scores数组的索引对应原始样本索引

    # 分析结果
    analyze_tracin_stability(args, all_scores_dict, selected_indices, n_samples, order_names)

def analyze_tracin_stability(args, all_scores_dict, selected_indices, n_samples, order_names):
    """分析选定的100个点在两种顺序下的TracIn分数稳定性"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing TracIn stability for selected samples")

    if len(order_names) != 2:
        logger.error("Expected two orders (selected_first, selected_last) for stability analysis.")
        return

    scores_first = all_scores_dict[order_names[0]] # 对应 selected_first
    scores_last = all_scores_dict[order_names[1]]  # 对应 selected_last

    # 提取选定样本的分数
    selected_scores_first = scores_first[selected_indices]
    selected_scores_last = scores_last[selected_indices]

    # 创建DataFrame进行比较
    comparison_df = pd.DataFrame({
        "sample_original_index": selected_indices,
        f"score_{order_names[0]}": selected_scores_first,
        f"score_{order_names[1]}": selected_scores_last
    })
    comparison_df["score_diff"] = comparison_df[f"score_{order_names[0]}"] - comparison_df[f"score_{order_names[1]}"]
    comparison_df["score_abs_diff"] = np.abs(comparison_df["score_diff"])
    
    # 计算均值和标准差等统计量 (可选)
    mean_score_first = np.mean(selected_scores_first)
    std_score_first = np.std(selected_scores_first)
    mean_score_last = np.mean(selected_scores_last)
    std_score_last = np.std(selected_scores_last)
    mean_abs_diff = np.mean(comparison_df["score_abs_diff"])
    
    logger.info(f"Stats for {len(selected_indices)} selected samples:")
    logger.info(f"Order '{order_names[0]}': Mean Score = {mean_score_first:.4f}, Std Score = {std_score_first:.4f}")
    logger.info(f"Order '{order_names[1]}': Mean Score = {mean_score_last:.4f}, Std Score = {std_score_last:.4f}")
    logger.info(f"Mean Absolute Difference between scores: {mean_abs_diff:.4f}")

    # 保存比较结果
    comparison_df.to_csv(os.path.join(args.save_path, "tracin_selected_samples_stability.csv"), index=False)
    logger.info(f"Stability analysis for selected samples saved to {os.path.join(args.save_path, 'tracin_selected_samples_stability.csv')}")

    # 可视化 (示例：散点图比较分数)
    plt.figure(figsize=(10, 8))
    plt.scatter(selected_scores_first, selected_scores_last, alpha=0.6)
    min_val = min(selected_scores_first.min(), selected_scores_last.min())
    max_val = max(selected_scores_first.max(), selected_scores_last.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="y=x (Perfect Stability)")
    plt.xlabel(f"TracIn Scores (Selected Samples in First Batch - {order_names[0]})")
    plt.ylabel(f"TracIn Scores (Selected Samples in Last Batch - {order_names[1]})")
    plt.title(f"TracIn Score Stability for {len(selected_indices)} Selected Samples")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_selected_samples_stability_scatter.png"))
    logger.info(f"Stability scatter plot saved to {os.path.join(args.save_path, 'tracin_selected_samples_stability_scatter.png')}")

    # 可视化分数差异的分布
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df["score_abs_diff"], kde=True)
    plt.xlabel("Absolute Difference in TracIn Scores")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Absolute Score Differences for {len(selected_indices)} Selected Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_selected_samples_score_diff_dist.png"))
    logger.info(f"Score difference distribution plot saved to {os.path.join(args.save_path, 'tracin_selected_samples_score_diff_dist.png')}")

# 替换旧的 analyze_tracin_order_results
# def analyze_tracin_order_results(args, n_orders, all_scores, n_samples):
# // ... existing code ...
#     logger.info(f"最大分数差异与平均值比率超过1的样本比例: {(pivot_df['range_ratio'] > 1).mean()}")
# // ... existing code ...

# 替换旧的 generate_different_orders
# def generate_different_orders(n_samples, n_orders=5, seed=42):
#     """生成多种不同的训练顺序"""
#     np.random.seed(seed)
#     orders = []
#
#     # 顺序1: 原始顺序
#     orders.append(np.arange(n_samples))
#
#     # 顺序2: 完全随机
#     random_order = np.arange(n_samples)
#     np.random.shuffle(random_order)
#     orders.append(random_order)
#
#     # 顺序3-n: 其他随机顺序
#     for i in range(n_orders - 2):
#         new_order = np.arange(n_samples)
#         np.random.seed(seed + i + 1)
#         np.random.shuffle(new_order)
#         orders.append(new_order)
#
#     return orders
