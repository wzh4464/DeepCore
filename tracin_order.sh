#!/bin/bash
# TracIn order experiment script

# 设置变量
DATASET="MNIST"
MODEL="LeNet"
SELECTION_METHOD="TracIn"
NUM_EXP=5
SELECTION_EPOCHS=5
NUM_SCORES=100
BATCH_SIZE=32
GPU_ID=0
DATA_PATH="./data"
SAVE_PATH="./results/tracin_order_experiment"

# 创建保存目录
mkdir -p "$SAVE_PATH"

# 运行实验
python main.py \
  --exp tracin_order \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --selection "$SELECTION_METHOD" \
  --num_exp "$NUM_EXP" \
  --selection_epochs "$SELECTION_EPOCHS" \
  --num_scores "$NUM_SCORES" \
  --batch "$BATCH_SIZE" \
  --gpu "$GPU_ID" \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH"

echo "实验完成！结果保存在: $SAVE_PATH" 