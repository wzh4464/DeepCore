import os
import torch
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Sampler, Subset
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

def generate_specific_orders(selected_indices, focus_indices, all_indices):
    """
    生成两种特定的训练顺序:
    1. 选定的 focus_indices 个样本在最前面，其余 selected_indices 在后面.
    2. 选定的 focus_indices 个样本在最后面，其余 selected_indices 在前面.
    
    参数:
    - selected_indices: 200个选定进行评估的样本的索引
    - focus_indices: 40个重点样本的索引（在selected_indices中）
    - all_indices: 所有样本的索引
    
    返回:
    - orders: 两种不同顺序的训练数据索引
    - order_names: 两种顺序的名称
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在生成特定训练顺序...")
    logger.info(f"选定的200个样本范围: {min(selected_indices)}-{max(selected_indices)}")
    logger.info(f"选定的40个焦点样本范围: {min(focus_indices)}-{max(focus_indices)}")
    logger.info(f"焦点样本占总体样本的 {len(focus_indices)/len(all_indices)*100:.2f}%, 占选定样本的 {len(focus_indices)/len(selected_indices)*100:.2f}%")

    # 确认focus_indices是selected_indices的子集
    if not set(focus_indices).issubset(set(selected_indices)):
        logger.error("错误: focus_indices不是selected_indices的子集!")
        raise ValueError("focus_indices必须是selected_indices的子集")

    # 获取selected_indices中不在focus_indices中的索引
    other_selected = [idx for idx in selected_indices if idx not in focus_indices]
    logger.info(f"其余160个非焦点样本范围: {min(other_selected)}-{max(other_selected)}")

    # 获取不在selected_indices中的其他所有样本索引
    remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
    logger.info(f"剩余未选定样本数量: {len(remaining_indices)}")

    # 生成两种不同的顺序
    # 顺序1: focus_indices在前，其他selected在后，剩余样本在最后
    order_focus_first = np.concatenate([focus_indices, other_selected, remaining_indices])
    
    # 顺序2: 其他selected在前，focus_indices在后，剩余样本在最后
    order_focus_last = np.concatenate([other_selected, focus_indices, remaining_indices])
    
    orders = [order_focus_first, order_focus_last]
    order_names = ["focus_first", "focus_last"]
    
    logger.info(f"已生成两种训练顺序: {order_names}")
    logger.info(f"顺序1 (focus_first): 40个焦点样本在前, 其余160个选定样本在中间, 剩余样本在最后")
    logger.info(f"顺序2 (focus_last): 160个非焦点样本在前, 40个焦点样本在中间, 剩余样本在最后")
    
    return orders, order_names

def run(args, checkpoint, start_exp, start_epoch):
    """运行TracIn顺序实验"""
    logger = logging.getLogger(__name__)
    logger.info("============= 开始运行 TracIn 顺序稳定性实验 =============")
    logger.info(f"参数: dataset={args.dataset}, model={args.model}, selection_epochs={args.selection_epochs}, seed={args.seed}")
    logger.info(f"实验目标: 验证TracIn对训练顺序的稳定性")

    # 加载数据集
    logger.info("正在加载数据集...")
    train_loader, test_loader, if_weighted, subset, selection_args = initialize_dataset_and_model(args, checkpoint)

    # 获取数据集
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    logger.info(f"数据集加载完成. 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 获取样本数量
    n_samples = len(train_dataset)
    all_indices = np.arange(n_samples)
    logger.info(f"数据集共有 {n_samples} 个样本")

    # 定义要选择的样本数量
    num_selected_samples = 200  # 选200个点
    num_focus_samples = 40      # 重点关注40个点
    logger.info(f"将从总样本中选择 {num_selected_samples} 个样本进行分析，并从中选择 {num_focus_samples} 个焦点样本")

    # 随机选择200个样本
    np.random.seed(args.seed)
    logger.info(f"使用随机种子 {args.seed} 选择样本")
    selected_indices = np.random.choice(all_indices, num_selected_samples, replace=False)
    selected_indices = np.sort(selected_indices)  # 排序以保持一致性
    logger.info(f"已随机选择 {num_selected_samples} 个样本, 索引范围: {min(selected_indices)}-{max(selected_indices)}")
    
    # 从200个样本中随机选择40个重点关注的样本
    focus_indices = np.random.choice(selected_indices, num_focus_samples, replace=False)
    focus_indices = np.sort(focus_indices)  # 排序以保持一致性
    logger.info(f"已从选定样本中随机选择 {num_focus_samples} 个焦点样本, 索引范围: {min(focus_indices)}-{max(focus_indices)}")

    # 创建只包含选中200个样本的数据集
    selected_train_dataset = Subset(train_dataset, selected_indices)
    logger.info(f"创建选定样本子集完成, 子集大小: {len(selected_train_dataset)}")
    
    # 生成特定的训练顺序
    orders, order_names = generate_specific_orders(selected_indices, focus_indices, all_indices)
    n_orders = len(orders)  # 现在是2

    # 将indices保存到文件
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        indices_df = pd.DataFrame({
            "selected_indices": selected_indices,
            "is_focus": [idx in focus_indices for idx in selected_indices]
        })
        indices_df.to_csv(os.path.join(args.save_path, "selected_samples_indices.csv"), index=False)
        logger.info(f"已保存选定样本索引到: {os.path.join(args.save_path, 'selected_samples_indices.csv')}")
        logger.info(f"所有结果将保存到: {args.save_path}")

    # 存储所有顺序的分数
    all_scores_dict = {}  # 使用字典存储，键为 order_name

    # 对每种顺序运行实验
    for i, order_name in enumerate(order_names):
        current_order = orders[i]
        logger.info(f"================ 开始运行顺序实验 {i+1}/{n_orders}: {order_name} ================")
        logger.info(f"顺序特点: {'40个焦点样本在前' if order_name == 'focus_first' else '40个焦点样本在后'}")

        # 创建自定义数据加载器，确保按照指定顺序加载数据
        custom_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            sampler=OrderedSampler(current_order),
            num_workers=args.workers,
            pin_memory=True
        )
        logger.info(f"已创建自定义数据加载器, batch_size={args.batch}, workers={args.workers}")

        # 初始化TracIn方法，只计算选定200个样本的分数
        logger.info(f"初始化TracIn方法...")
        logger.info(f"参数: epochs={args.selection_epochs}, num_test_samples={args.num_scores}")
        tracin = SELECTION_METHODS["TracIn"](
            train_dataset,
            args,
            fraction=args.fraction,
            random_seed=args.seed + i,
            epochs=args.selection_epochs,
            specific_model=args.model,
            checkpoint_interval=1,
            num_test_samples=args.num_scores,
            dst_test=test_dataset,
        )

        # 使用当前顺序训练模型
        logger.info(f"开始使用顺序 '{order_name}' 训练模型...")
        tracin.train_loader = custom_train_loader  # 确保TracIn使用我们的自定义加载器
        tracin.before_run()
        tracin.run()
        logger.info(f"模型训练完成.")

        # 计算TracIn scores (只获取选定200个样本的分数)
        logger.info(f"开始计算TracIn分数...")
        all_scores = tracin.get_scores() 
        logger.info(f"计算完成, 分数形状: {all_scores.shape if hasattr(all_scores, 'shape') else '标量'}")
        
        # 转换为numpy数组
        if isinstance(all_scores, torch.Tensor):
            all_scores = all_scores.detach().cpu().numpy()
            logger.info(f"已将分数从Tensor转换为numpy数组")
            
        # 只保留选定的200个样本的分数
        scores = all_scores[selected_indices]
        logger.info(f"已提取选定的200个样本的分数, 分数范围: {np.min(scores):.6f} - {np.max(scores):.6f}")
        logger.info(f"分数统计: 平均值={np.mean(scores):.6f}, 标准差={np.std(scores):.6f}")

        # 保存分数
        scores_df = pd.DataFrame({
            "original_index": selected_indices,
            "tracin_score": scores,
            "is_focus": [idx in focus_indices for idx in selected_indices]
        })
        scores_df.to_csv(os.path.join(args.save_path, f"tracin_scores_{order_name}.csv"), index=False)
        logger.info(f"已保存200个选定样本的分数到: {os.path.join(args.save_path, f'tracin_scores_{order_name}.csv')}")

        all_scores_dict[order_name] = scores  # 只存储200个选定样本的分数

    # 分析结果
    logger.info("开始分析两种顺序下的TracIn分数稳定性...")
    analyze_tracin_stability(args, all_scores_dict, selected_indices, focus_indices, order_names)
    logger.info("============= TracIn 顺序稳定性实验完成 =============")

def analyze_tracin_stability(args, all_scores_dict, selected_indices, focus_indices, order_names):
    """分析选定的40个重点样本在两种顺序下的TracIn分数稳定性"""
    logger = logging.getLogger(__name__)
    logger.info("开始分析选定样本的TracIn分数稳定性...")

    if len(order_names) != 2:
        logger.error(f"错误: 预期有两种顺序, 实际有 {len(order_names)} 种")
        return

    # 获取两种顺序的所有200个选定样本的分数
    logger.info(f"获取两种顺序的分数...")
    scores_first = all_scores_dict[order_names[0]]  # 对应 focus_first
    scores_last = all_scores_dict[order_names[1]]   # 对应 focus_last
    logger.info(f"顺序1 '{order_names[0]}' 分数范围: {np.min(scores_first):.6f} - {np.max(scores_first):.6f}")
    logger.info(f"顺序2 '{order_names[1]}' 分数范围: {np.min(scores_last):.6f} - {np.max(scores_last):.6f}")

    # 为了保持数据一致性，构建一个映射从原始索引到选定200个样本中的位置
    selected_idx_map = {idx: i for i, idx in enumerate(selected_indices)}
    logger.info(f"创建了原始索引到200个样本位置的映射")

    # 获取40个焦点样本在200个样本中的位置索引
    focus_positions = [selected_idx_map[idx] for idx in focus_indices]
    logger.info(f"获取40个焦点样本在200个样本中的位置索引")

    # 提取40个焦点样本的分数
    focus_scores_first = scores_first[focus_positions]
    focus_scores_last = scores_last[focus_positions]
    logger.info(f"提取40个焦点样本的分数")
    logger.info(f"顺序1 '{order_names[0]}' 焦点样本分数: 平均值={np.mean(focus_scores_first):.6f}, 标准差={np.std(focus_scores_first):.6f}")
    logger.info(f"顺序2 '{order_names[1]}' 焦点样本分数: 平均值={np.mean(focus_scores_last):.6f}, 标准差={np.std(focus_scores_last):.6f}")

    # 创建DataFrame进行比较
    logger.info(f"创建分数比较DataFrame...")
    comparison_df = pd.DataFrame({
        "sample_original_index": focus_indices,
        f"score_{order_names[0]}": focus_scores_first,
        f"score_{order_names[1]}": focus_scores_last
    })
    comparison_df["score_diff"] = comparison_df[f"score_{order_names[0]}"] - comparison_df[f"score_{order_names[1]}"]
    comparison_df["score_abs_diff"] = np.abs(comparison_df["score_diff"])
    logger.info(f"计算分数差异完成")

    # 计算均值和标准差等统计量
    mean_score_first = np.mean(focus_scores_first)
    std_score_first = np.std(focus_scores_first)
    mean_score_last = np.mean(focus_scores_last)
    std_score_last = np.std(focus_scores_last)
    mean_abs_diff = np.mean(comparison_df["score_abs_diff"])
    max_abs_diff = np.max(comparison_df["score_abs_diff"])
    min_abs_diff = np.min(comparison_df["score_abs_diff"])
    median_abs_diff = np.median(comparison_df["score_abs_diff"])
    pearson_corr = np.corrcoef(focus_scores_first, focus_scores_last)[0, 1]
    
    logger.info(f"========== 统计结果: {len(focus_indices)} 个焦点样本 ==========")
    logger.info(f"顺序 '{order_names[0]}': 均值 = {mean_score_first:.4f}, 标准差 = {std_score_first:.4f}")
    logger.info(f"顺序 '{order_names[1]}': 均值 = {mean_score_last:.4f}, 标准差 = {std_score_last:.4f}")
    logger.info(f"分数绝对差异: 均值 = {mean_abs_diff:.4f}, 最大值 = {max_abs_diff:.4f}, 最小值 = {min_abs_diff:.4f}, 中位数 = {median_abs_diff:.4f}")
    logger.info(f"Pearson相关系数: {pearson_corr:.4f}")
    
    # 分数大小变化的样本比例
    score_direction_change = (focus_scores_first * focus_scores_last) < 0
    direction_change_ratio = np.mean(score_direction_change)
    logger.info(f"分数符号发生变化的样本比例: {direction_change_ratio:.2%}")
    
    # 分数排名变化分析
    rank_first = pd.Series(focus_scores_first).rank()
    rank_last = pd.Series(focus_scores_last).rank()
    rank_diff = np.abs(rank_first - rank_last)
    mean_rank_change = np.mean(rank_diff)
    logger.info(f"平均排名变化: {mean_rank_change:.2f} (满分40)")
    logger.info(f"排名变化超过样本总数20%的样本比例: {np.mean(rank_diff > len(focus_indices)*0.2):.2%}")

    # 保存比较结果
    comparison_df["rank_first"] = rank_first
    comparison_df["rank_last"] = rank_last
    comparison_df["rank_diff"] = rank_diff
    comparison_df["direction_change"] = score_direction_change
    comparison_df.to_csv(os.path.join(args.save_path, "tracin_focus_samples_stability.csv"), index=False)
    logger.info(f"已保存焦点样本稳定性分析结果到: {os.path.join(args.save_path, 'tracin_focus_samples_stability.csv')}")

    # 可视化 - 散点图比较分数
    logger.info(f"开始生成可视化图表...")
    plt.figure(figsize=(10, 8))
    plt.scatter(focus_scores_first, focus_scores_last, alpha=0.6)
    min_val = min(focus_scores_first.min(), focus_scores_last.min())
    max_val = max(focus_scores_first.max(), focus_scores_last.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="y=x (Perfect Stability)")
    plt.xlabel(f"TracIn Scores ({order_names[0]})")
    plt.ylabel(f"TracIn Scores ({order_names[1]})")
    plt.title(f"TracIn Score Stability for {len(focus_indices)} Focus Samples\nPearson Corr: {pearson_corr:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_focus_samples_stability_scatter.png"))
    logger.info(f"已生成散点图: {os.path.join(args.save_path, 'tracin_focus_samples_stability_scatter.png')}")

    # 可视化 - 40个样本在200样本中的分数分布情况
    plt.figure(figsize=(12, 8))

    # 排序分数以便绘制分布
    sorted_first = np.sort(scores_first)
    sorted_last = np.sort(scores_last)

    # 在200点的分数中标记40个焦点点的位置
    plt.plot(sorted_first, 'b-', alpha=0.7, label=f"{order_names[0]} - All 200 samples")
    plt.plot(sorted_last, 'g-', alpha=0.7, label=f"{order_names[1]} - All 200 samples")

    plt.scatter([np.searchsorted(sorted_first, score) for score in focus_scores_first], 
               focus_scores_first, color='red', marker='o', s=50, label=f"{order_names[0]} - 40 focus samples")
    plt.scatter([np.searchsorted(sorted_last, score) for score in focus_scores_last], 
               focus_scores_last, color='orange', marker='x', s=50, label=f"{order_names[1]} - 40 focus samples")

    plt.xlabel("Sample Index (Sorted by Score)")
    plt.ylabel("TracIn Score")
    plt.title("Distribution of 40 Focus Samples in 200 Selected Samples")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_focus_samples_distribution.png"))
    logger.info(f"已生成分布图: {os.path.join(args.save_path, 'tracin_focus_samples_distribution.png')}")

    # 可视化分数差异的分布
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df["score_abs_diff"], kde=True)
    plt.xlabel("Absolute Difference in TracIn Scores")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Absolute Score Differences\nMean: {mean_abs_diff:.4f}, Max: {max_abs_diff:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_focus_samples_score_diff_dist.png"))
    logger.info(f"已生成差异分布图: {os.path.join(args.save_path, 'tracin_focus_samples_score_diff_dist.png')}")
    
    # 可视化排名变化
    plt.figure(figsize=(10, 6))
    sns.histplot(rank_diff, kde=True, bins=20)
    plt.axvline(x=mean_rank_change, color='r', linestyle='--', label=f"Mean Rank Change: {mean_rank_change:.2f}")
    plt.xlabel("Absolute Rank Change")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Rank Changes for 40 Focus Samples")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tracin_focus_samples_rank_change.png"))
    logger.info(f"已生成排名变化图: {os.path.join(args.save_path, 'tracin_focus_samples_rank_change.png')}")
    
    logger.info(f"全部可视化分析完成")
