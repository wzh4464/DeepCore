# TracIn顺序实验

本实验旨在验证不同训练顺序对TracIn分数的影响，通过控制训练样本的顺序，分析在不同顺序下计算得到的TracIn分数的变化情况。

## 功能说明

实验主要包含以下功能：

1. 生成多种不同的训练样本顺序
2. 对每种顺序训练模型并计算TracIn分数
3. 分析同一样本在不同顺序下的分数差异
4. 可视化结果和统计分析

## 使用方法

### 直接运行脚本

最简单的方法是直接运行提供的脚本：

```bash
./tracin_order.sh
```

### 自定义参数运行

也可以通过直接调用Python脚本并指定参数来运行：

```bash
python main.py \
  --exp tracin_order \
  --dataset MNIST \
  --model LeNet \
  --selection TracIn \
  --num_exp 5 \
  --selection_epochs 5 \
  --num_scores 100 \
  --batch 32 \
  --gpu 0 \
  --data_path ./data \
  --save_path ./results/tracin_order_experiment
```

参数说明：
- `--exp tracin_order`：指定运行TracIn顺序实验
- `--num_exp 5`：生成5种不同的训练顺序
- `--selection_epochs 5`：每种顺序训练5个epoch
- `--num_scores 100`：用于评估的测试样本数量
- `--batch 32`：训练的批量大小

## 结果分析

实验完成后，会在`--save_path`指定的目录中生成以下结果：

1. 每种顺序的训练样本顺序：`order_0.csv`, `order_1.csv`, ...
2. 每种顺序下的TracIn分数：`tracin_scores_order_0.csv`, `tracin_scores_order_1.csv`, ...
3. 所有分数的汇总：`all_tracin_scores.csv`
4. 统计分析结果：`tracin_scores_stats.csv`
5. 可视化图表：`tracin_order_analysis.png`

### 重要统计指标

实验会计算以下统计指标来评估训练顺序对TracIn分数的影响：

- **变异系数(CV)**：标准差除以平均值的绝对值，表示分数的相对变异程度
- **范围比率**：最大值与最小值的差除以平均值的绝对值，表示最大变化范围相对于平均分数的比例

如果变异系数和范围比率较大（如超过0.5或1），则表明训练顺序对TracIn分数有显著影响。

### 结果解读

实验输出日志会提供以下统计信息：
- 平均标准差：所有样本在不同顺序下分数标准差的平均值
- 最大标准差：所有样本中最大的标准差值
- 平均变异系数：所有样本变异系数的平均值
- 变异系数超过0.5的样本比例：反映显著受顺序影响的样本比例
- 最大分数差异与平均值比率超过1的样本比例：反映极端受顺序影响的样本比例

## 注意事项

1. 实验需要较大的计算资源和存储空间，特别是当数据集较大或顺序数量较多时
2. 建议首先在小数据集上运行测试，如MNIST，然后再扩展到更大的数据集
3. 增加`--num_exp`参数可以获得更多的顺序样本，但也会增加实验时间 