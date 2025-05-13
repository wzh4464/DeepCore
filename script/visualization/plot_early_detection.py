import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
import sys

# 添加项目根目录到系统路径，以便引入experiment包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiment.experiment_utils import PlotManager  # 引入PlotManager

# 配置参数
METHODS = ["oti", "grand", "influence_function"]  # 所有方法
FLIP_NUMS = [10, 20, 30, 40]  # 所有flip数目
EPOCHS = [1, 2, 3, 4, 5]  # 所有epoch数
SEEDS = list(range(8))  # 所有种子 0-7
LR = 0.05  # 学习率

# 方法名映射
METHOD_NAMES = {
    "oti": "OTI",
    "grand": "GraNd",
    "influence_function": "Influence Function"
}

def count_found_flipped_samples(file_path):
    """
    读取found_flipped_indices文件，统计找到的翻转样本数量
    
    参数:
        file_path: 文件路径，例如: "results/early_detection/oti/nf10_lr0.05_seed0/OTI_found_flipped_indices_epochs1_0.csv"
    
    返回:
        int: 找到的翻转样本数量
    """
    try:
        df = pd.read_csv(file_path)
        if 'found_flipped_indices' in df.columns:
            # 对非空值进行计数，处理可能的NaN
            return df['found_flipped_indices'].count()
        else:
            # 如果格式不对，直接计算行数（减去表头）
            return len(df) - 1 if len(df) > 0 else 0
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

def collect_flipped_stats():
    """
    收集所有方法、所有flip数目、所有epoch的翻转样本检测数量
    
    遍历results/early_detection目录下的所有实验结果，
    统计各个方法在不同翻转样本数量和不同训练轮次下检测到的翻转样本数量
    
    返回:
        dict: 嵌套字典，格式为{flip_num: {method: {epoch: [counts...]}}}
              例如stats[10]["oti"][1]表示初始翻转数为10时，OTI方法在第1轮训练后检测到的翻转样本数列表
    """
    # 三层嵌套字典，用于存储统计结果
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 遍历所有方法
    for method in METHODS:
        base_dir = f"results/early_detection/{method}"
        if not os.path.exists(base_dir):
            print(f"警告: 目录 {base_dir} 不存在")
            continue
            
        # 遍历所有flip数目和种子组合
        for flip_num in FLIP_NUMS:
            for seed in SEEDS:
                # 构建目录路径
                result_dir = f"{base_dir}/nf{flip_num}_lr{LR}_seed{seed}"
                if not os.path.exists(result_dir):
                    print(f"警告: 目录 {result_dir} 不存在")
                    continue
                
                # 读取每个epoch的结果
                for epoch in EPOCHS:
                    # 构建文件路径，匹配实验中对应的方法文件
                    if method == "oti":
                        method_name = "OTI"
                    elif method == "grand":
                        method_name = "GraNd"
                    else:
                        method_name = "influence_function"
                        
                    file_path = f"{result_dir}/{method_name}_found_flipped_indices_epochs{epoch}_0.csv"
                    if not os.path.exists(file_path):
                        print(f"警告: 文件 {file_path} 不存在")
                        continue
                    
                    # 统计找到的翻转样本数
                    count = count_found_flipped_samples(file_path)
                    stats[flip_num][method][epoch].append(count)
    
    return stats

def plot_flipped_stats(stats):
    """
    使用PlotManager为每个flip数目绘制一张图，只显示每个方法随epoch变化的平均检测数量
    
    参数:
        stats: 由collect_flipped_stats()返回的统计数据
    
    生成的图表将保存在results/early_detection目录下，文件名格式为：
    flipped_detection_nf{flip_num}_{timestamp}.png
    
    每张图表显示不同方法在不同训练轮次下检测到的翻转样本平均数量
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 为每个flip数目创建一个图
    for flip_num in FLIP_NUMS:
        pm = PlotManager(
            fig_size=(10, 6),
            font_size=18,
            axes_titlesize=18,
            axes_labelsize=16,
            xtick_labelsize=14,
            ytick_labelsize=14,
            legend_fontsize=14
        )
        
        # 绘制每个方法的曲线
        for method in METHODS:
            if method not in stats[flip_num]:
                continue
                
            # 计算每个epoch的平均值
            means = []
            for epoch in EPOCHS:
                if epoch in stats[flip_num][method] and len(stats[flip_num][method][epoch]) > 0:
                    means.append(np.mean(stats[flip_num][method][epoch]))
                else:
                    means.append(0)
            
            # 使用PlotManager的plot方法绘制平均值曲线
            label = METHOD_NAMES.get(method, method)
            pm.plot(EPOCHS, means, label=label, marker='o', linewidth=3)
        
        # 设置图表属性
        pm.set_title(f'Initial Flipped Samples: {flip_num}')
        pm.set_xlabel('Training Epochs')
        pm.set_ylabel('Number of Detected Flipped Samples')
        pm.set_xticks_int()
        pm.set_xlim(0.5, 5.5)
        pm.add_grid()
        pm.add_legend(loc='upper left')
        
        # 保存图表
        save_path = f"results/early_detection/flipped_detection_nf{flip_num}_{timestamp}.png"
        pm.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        pm.close()

def main():
    """
    主函数：收集并绘制各方法在不同翻转样本数量和训练轮次下的检测性能
    """
    # 确保结果目录存在
    if not os.path.exists("results/early_detection"):
        print("错误: results/early_detection 目录不存在")
        return

    # 收集统计数据
    print("正在收集数据统计...")
    stats = collect_flipped_stats()

    # 生成图表
    print("正在生成图表...")
    plot_flipped_stats(stats)

    print("所有图表生成完成!")

if __name__ == "__main__":
    main() 
