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

```
.
├── main.py              # Main entry point
├── deepcore/
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
