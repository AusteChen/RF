#!/bin/bash
# ========================================
# V2 Rolling Forcing Inference Script
# ========================================

# 配置参数
CONFIG_PATH="/home/adminad/GaoyanTian/code/RollingForcing/configs/exp0_curriculum_linear.yaml"
CHECKPOINT_PATH="/home/adminad/GaoyanTian/model/rolling_forcing/exp0a/checkpoint_latest.pt"
DATA_PATH="/home/adminad/GaoyanTian/code/RollingForcing/prompts/prompt.txt"
OUTPUT_FOLDER="/home/adminad/GaoyanTian/tmp/rf-videos/inference_v2"

# 推理参数
NUM_OUTPUT_FRAMES=81
NUM_SAMPLES=1
SEED=0

# ========================================
# V2 架构参数（可选）
# ========================================
# 方式1: 命令行覆盖配置文件
# USE_DUAL_CHANNEL_HEAD="--use_dual_channel_head true"
# USE_GUMBEL_ROUTER="--use_gumbel_router true"
# COMPRESSION_RATIO="--compression_ratio 4"
# GLOBAL_LAYER_INDICES='--global_layer_indices "0,5,10,15"'

# 方式2: 不设置参数，使用配置文件中定义的值
# 或者使用空字符串（将使用配置文件默认值）
USE_DUAL_CHANNEL_HEAD=""
USE_GUMBEL_ROUTER=""
COMPRESSION_RATIO=""
GLOBAL_LAYER_INDICES=""

# 创建输出目录
mkdir -p "$OUTPUT_FOLDER"

# 运行推理
torchrun --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  inference_v2.py -- \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_output_frames $NUM_OUTPUT_FRAMES \
  --num_samples $NUM_SAMPLES \
  --seed $SEED \
  $USE_DUAL_CHANNEL_HEAD \
  $USE_GUMBEL_ROUTER \
  $COMPRESSION_RATIO \
  $GLOBAL_LAYER_INDICES \
  2>&1 | tee "$OUTPUT_FOLDER/inference.log"
