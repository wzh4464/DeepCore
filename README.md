# LiveVal: Time-aware Data Valuation for SGD-trained Models via Adaptive Reference Points

This repository contains the official implementation of LiveVal, a novel approach for time-aware data valuation in SGD-trained models. LiveVal introduces an adaptive reference points mechanism that dynamically evaluates the importance of training samples during the model training process. It also includes various experiment modes for evaluating data valuation methods.

## Overview

LiveVal extends the Online Training Influence (OTI) method by incorporating time-aware data valuation through adaptive reference points. This approach allows for more accurate assessment of training samples' contributions to model performance. The project supports multiple experiment modes including standard training/evaluation, label flipping (`flip`), input corruption (`corrupt`), early detection of mislabeled samples (`early_detection`), and boundary sample detection (`boundary_detection`).

## Key Features

- **Adaptive Reference Points**: Dynamically adjusts reference points based on loss changes during training for methods like AD_OTI.
- **Time-Aware Valuation**: Considers temporal aspects of training dynamics.
- **Multiple Experiment Modes**: Supports `train_and_eval`, `flip`, `corrupt`, `early_detection`, `boundary_detection`.
- **Various Data Valuation Methods**: Implements methods like `OTI`, `GraNd`, `influence_function`, `uniform`, `forgetting`, etc. (defined in `liveval/methods/selection_methods.py`).
- **Memory-Efficient**: Optimized implementation for handling large-scale datasets.
- **Multiple Dataset Support**: Works with various datasets including MNIST, CIFAR10, and custom datasets.
- **Flexible Model Architecture**: Compatible with different neural network architectures.

## Installation

```bash
# Clone the repository
git clone https://github.com/LiveVal/LiveVal.git
cd LiveVal

# Create and activate a virtual environment (example using conda)
conda create -p $PWD/.venv python=3.12
conda activate $PWD/.venv

# Install dependencies (check requirements.txt or spec-file.txt)
pip install -r requirements.txt
# Or if using conda environment file:
# conda install -p $PWD/.venv --file spec-file.txt
```

## Usage

### Basic Training and Evaluation (`train_and_eval`)

```bash
python main.py \
    --exp train_and_eval \
    --dataset CIFAR10 \
    --model ResNet18 \
    --selection OTI \
    --num_exp 1 \
    --epochs 200 \
    --selection_epochs 40 \
    --data_path ./data \
    --gpu 0 \
    --fraction 0.1 \
    --optimizer SGD \
    --lr 0.1 \
    --scheduler CosineAnnealingLR \
    --save_path ./results/train_eval \
    --num_gpus 1
```

### Label Flipping Experiment (`flip`)

Calculates scores for a dataset with a specified number of flipped labels.

```bash
python main.py \
    --exp flip \
    --dataset CIFAR10 \
    --model ResNet18 \
    --selection OTI \
    --num_exp 5 \
    --epochs 200 \
    --selection_epochs 40 \
    --data_path ./data \
    --gpu 0 \
    --num_flip 100 \
    --save_path ./results/flip_exp \
    --num_gpus 1
```

### Early Detection Experiment (`early_detection`)

Compares methods' ability to detect flipped samples early in training.

```bash
python main.py \
    --exp early_detection \
    --dataset CIFAR10 \
    --model ResNet18 \
    --selection OTI \
    --num_exp 1 \
    --epochs 20 \
    --selection_epochs 5 \
    --data_path ./data \
    --gpu 0 \
    --num_flip 100 \
    --save_path ./results/early_detection_exp \
    --num_gpus 1
```

(Run separately for each method like `GraNd`, `influence_function`)

### Other Experiments

- **`corrupt`**: Corrupts a number of inputs. Use `--num_corrupt`.
- **`boundary_detection`**: Detects boundary samples. Use `--num_boundary` and `--boundary_transform_intensity`.

### Key Parameters for Adaptive OTI (AD_OTI)

```bash
python main.py \
    --selection AD_OTI \
    --delta_min 1 \
    --delta_max 20 \
    --delta_step 3 \
    --eps_min 0.005 \
    --eps_max 0.01 \
    --oti_mode full \
    # ... other common parameters
```

### Using SLURM

The repository may include SLURM job scripts (e.g., `slurm_job.sh`). Adapt the script arguments as needed.

```bash
# Example structure (adapt based on actual script)
sbatch slurm_job.sh \\
    --exp early_detection \\
    --selection OTI \\
    --dataset CIFAR10 \\
    --model ResNet18 \\
    --num_flip 100 \\
    --epochs 20 \\
    --selection_epochs 5 \\
    --lr 0.1 \\
    --seed 42 \\
    --save_path ./results/slurm_early_detection \\
    --data_path /path/to/datasets \\
    # ... other relevant SLURM script arguments (e.g., scheduler, optimizer, batch size)
```

## Key Parameters

- `--exp`: Experiment mode (`train_and_eval`, `flip`, `corrupt`, `early_detection`, `boundary_detection`).
- `--selection`: Data valuation/selection method (e.g., `OTI`, `AD_OTI`, `GraNd`, `uniform`, `influence_function`).
- `--num_flip`: Number of labels to flip in `flip` and `early_detection` modes.
- `--num_corrupt`: Number of inputs to corrupt in `corrupt` mode.
- `--num_boundary`: Number of boundary points to generate in `boundary_detection` mode.
- `--boundary_transform_intensity`: Intensity of transformation for boundary points.
- `--delta_min`: (AD_OTI) Minimum window size for reference points.
- `--delta_max`: (AD_OTI) Maximum window size for reference points.
- `--delta_step`: (AD_OTI) Step size for window adjustment.
- `--eps_min`: (AD_OTI) Lower threshold for loss change.
- `--eps_max`: (AD_OTI) Upper threshold for loss change.
- `--oti_mode`: OTI operation mode (`full`, `stored`, `scores`).
- `--num_scores`: (LOO) Number of scores to calculate.
- *Refer to `main.py` `parse_args()` for a complete list of parameters.*

## Project Structure

```bash
.
├── main.py                 # Main entry point
├── README.md
├── requirements.txt        # Dependencies
├── liveval/                # Core library
│   ├── methods/            # Data valuation/selection methods (OTI, GraNd, etc.)
│   ├── datasets/           # Dataset handling (CIFAR, MNIST, Flipped, Corrupted)
│   ├── nets/               # Network architectures (ResNet, LeNet)
│   └── utils/              # Utility functions
├── experiment/             # Experiment logic implementations
│   ├── train_and_eval.py
│   ├── flip.py
│   ├── corrupt.py
│   ├── early_detection_comparison.py
│   ├── boundary_detection.py
│   └── experiment_utils.py # Shared experiment utilities
├── script/                 # Scripts for running specific experiments/tasks
│   ├── early_detection_experiment.py # Example script to run early detection
│   └── ...
├── tests/                  # Unit tests
├── results/                # Default directory for saving experiment outputs
├── data/                   # Default directory for datasets
└── logs/                   # Default directory for logs
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon the original OTI method and extends it with adaptive reference points and various experimental setups for data valuation research.

## 实验结果文件说明

实验结果通常保存在 `--save_path` 指定的目录下，子目录结构可能因实验类型 (`--exp`) 和参数而异。

以 `early_detection` 实验为例 (`python main.py --exp early_detection --save_path ./results/early_detection_exp ...`)，结果可能保存在 `./results/early_detection_exp/` 下。其内部可能包含：

- `early_detection_results_时间戳.csv`：主结果表，记录各方法在不同 `selection_epochs` 下的检测率、移除翻转样本前后的准确率等。
- `accuracy_vs_detection_时间戳.png`, `detection_rate_vs_epochs_时间戳.png` 等：实验过程和结果的可视化图表。
- `[方法名]_scores_epochsX_expY.csv`：指定方法 (`--selection`) 在第 X 个 `selection_epoch`、第 Y 次实验 (`--num_exp`) 时计算出的样本分数。
- `[方法名]_epoch_accuracies_epochsX_expY.csv`：重新训练期间每个 epoch 的准确率。
- `[方法名]_step_losses_epochsX_expY.csv`：重新训练期间每步的 loss 记录。
- `[方法名]_found_flipped_indices_epochsX_expY.csv`：检测到的翻转样本索引。
- `flipped_selection_from.csv`: 记录了本次实验实际翻转的样本信息。

对于 `flip` 实验 (`python main.py --exp flip --save_path ./results/flip_exp ...`)，结果可能保存在 `./results/flip_exp/` 下，可能包含：

- `flip_scores_expY.csv`: 第 Y 次实验计算出的样本分数。
- `average_score.csv`: 多次实验 (`--num_exp`) 分数的平均值，包含样本索引和原始标签。
