# LiveVal: Time-aware Data Valuation for SGD-trained Models via Adaptive Reference Points

This repository contains the official implementation of LiveVal, a novel approach for time-aware data valuation in SGD-trained models. LiveVal introduces an adaptive reference points mechanism that dynamically evaluates the importance of training samples during the model training process.

## Overview

LiveVal extends the Online Training Influence (OTI) method by incorporating time-aware data valuation through adaptive reference points. This approach allows for more accurate assessment of training samples' contributions to model performance.

## Key Features

- **Adaptive Reference Points**: Dynamically adjusts reference points based on loss changes during training
- **Time-Aware Valuation**: Considers temporal aspects of training dynamics
- **Memory-Efficient**: Optimized implementation for handling large-scale datasets
- **Multiple Dataset Support**: Works with various datasets including MNIST, CIFAR10, and custom datasets
- **Flexible Model Architecture**: Compatible with different neural network architectures

## Installation

```bash
# Clone the repository
git clone https://github.com/LiveVal/LiveVal.git
cd LiveVal

# Create and activate a virtual environment
conda create -p $PWD/.venv python=3.12
conda activate $PWD/.venv

# Install dependencies
conda install -p $PWD/.venv --file spec-file.txt
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main.py \
    --dataset MNIST \
    --model LeNet \
    --selection AD_OTI \
    --num_exp 1 \
    --epochs 5 \
    --selection_epochs 5 \
    --data_path ./data \
    --gpu 0 \
    --optimizer SGD \
    --lr 0.1 \
    --scheduler CosineAnnealingLR \
    --save_path ./results \
    --num_gpus 1
```

### Advanced Configuration

LiveVal supports various hyperparameters for fine-tuning the adaptive reference points:

```bash
python main.py \
    --delta_min 1 \
    --delta_max 20 \
    --delta_step 3 \
    --eps_min 0.005 \
    --eps_max 0.01 \
    --oti_mode full
```

### Using SLURM

The repository includes SLURM job scripts for cluster environments:

```bash
sbatch slurm_job.sh \
    --mode AD_OTI \
    --regularization 1 \
    --learning_rate 1 \
    --adoti_mode full \
    --delta_min 1 \
    --delta_max 20 \
    --delta_step 3 \
    --eps_min 0.005 \
    --eps_max 0.01
```

## Key Parameters

- `delta_min`: Minimum window size for reference points (default: 1)
- `delta_max`: Maximum window size for reference points (default: 3)
- `delta_step`: Step size for window adjustment (default: 1)
- `eps_min`: Lower threshold for loss change (default: 0.1)
- `eps_max`: Upper threshold for loss change (default: 0.05)
- `oti_mode`: Operation mode ['full', 'stored', 'scores']

## Project Structure

```bash
.
├── main.py              # Main entry point
├── liveval/
│   ├── methods/
│   │   ├── oti.py      # Base OTI implementation
│   │   ├── ad_oti.py   # Adaptive OTI implementation
│   │   └── ...
│   ├── datasets/
│   │   └── ...         # Dataset implementations
│   └── nets/
│       └── ...         # Neural network architectures
└── utils/              # Utility functions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon the original OTI method and extends it with adaptive reference points for time-aware data valuation.

## 实验结果文件说明

以 `python -m script.early_detection_experiment` 运行早期检测实验为例，所有结果将自动保存在 `results/early_detection/` 目录下，按方法（如 grand、influence_function、oti）和参数（如 nf、lr、seed）分文件夹组织。例如：

```bash
results/early_detection/grand/nf10_lr0.05_seed0/
results/early_detection/oti/nf20_lr0.05_seed3/
results/early_detection/influence_function/nf40_lr0.05_seed7/
```

每个实验文件夹下包含如下类型的结果文件：

- `early_detection_results_时间戳.csv`：主结果表，记录每轮检测的主要指标（如检测率、准确率等）。
- `accuracy_vs_detection_时间戳.png`、`accuracy_improvement_vs_epochs_时间戳.png`、`detection_rate_vs_epochs_时间戳.png`：实验过程和结果的可视化图表。
- `[方法名]_scores_epochsX_0.csv`：每个 epoch 的样本分数（如 OTI_scores_epochs5_0.csv、GraNd_scores_epochs5_0.csv）。
- `[方法名]_epoch_accuracies_epochsX_0.csv`：每个 epoch 的准确率。
- `[方法名]_step_losses_epochsX_0.csv`：每步的 loss 记录。
- `[方法名]_found_flipped_indices_epochsX_0.csv`：每轮检测中被 flip 的样本索引。
- `flipped_indices.csv`、`flipped_selection_from.csv`：最终被 flip 的样本索引及来源。
- `permuted_indices.csv`：样本顺序的置换信息。
- `initial_params.pt`、`best_params.pkl`、`influence_model.pt` 等：模型参数快照。
- `epoch_0_data.pkl`：首轮数据快照。

不同 selection 方法（如 OTI、GraNd、influence_function）均会生成上述结构的结果文件，便于横向对比。

### 结果文件举例

以 `results/early_detection/oti/nf10_lr0.05_seed0/` 为例，主要文件说明如下：

- `early_detection_results_2025-05-12_23-07-41.csv`：主结果表，含各 epoch 检测率、准确率等。
- `OTI_scores_epochs5_0.csv`：第5个 epoch 的 OTI 分数。
- `OTI_found_flipped_indices_epochs5_0.csv`：第5个 epoch 检测到的 flip 样本索引。
- `accuracy_vs_detection_2025-05-12_23-07-41.png`：准确率与检测率关系图。
- `initial_params.pt`：初始模型参数。
- `flipped_indices.csv`：最终 flip 样本索引。

其余方法（如 GraNd、influence_function）命名方式类似。
