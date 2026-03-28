#!/bin/bash
# ========================================
# V2 Rolling Forcing Inference Script
# ========================================

# 配置参数
CONFIG_PATH="/home/adminad/GaoyanTian/code/RollingForcing/configs/exp0_linear.yaml"
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
# LOCAL_ATTN_SIZE="--local_attn_size 1"
# SINK_SIZE="--sink_size 1"
# USE_DYNAMIC_ANCHOR="--use_dynamic_anchor true"
# USE_LAYER_SPECIALIZATION="--use_layer_specialization true"
# USE_AUTO_LAYER_CLASSIFICATION="--use_auto_layer_classification true"
# GLOBAL_LAYER_INDICES='--global_layer_indices "0,5,10,15"'
# LOCAL_HISTORY_BLOCKS="--local_history_blocks 2"
# ANCHOR_BLOCKS="--anchor_blocks 1"
# SCENE_CHANGE_TAU="--scene_change_tau 0.6"

# 方式2: 不设置参数，使用配置文件中定义的值
# 或者使用空字符串（将使用配置文件默认值）
USE_DYNAMIC_ANCHOR=""
USE_LAYER_SPECIALIZATION=""
USE_AUTO_LAYER_CLASSIFICATION=""
GLOBAL_LAYER_INDICES=""
LOCAL_HISTORY_BLOCKS=""
ANCHOR_BLOCKS=""
SCENE_CHANGE_TAU=""

# 创建输出目录
mkdir -p "$OUTPUT_FOLDER"

# 运行推理
torchrun --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  inference.py -- \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_output_frames $NUM_OUTPUT_FRAMES \
  --num_samples $NUM_SAMPLES \
  --seed $SEED \
  $USE_DYNAMIC_ANCHOR \
  $USE_LAYER_SPECIALIZATION \
  $USE_AUTO_LAYER_CLASSIFICATION \
  $GLOBAL_LAYER_INDICES \
  $LOCAL_HISTORY_BLOCKS \
  $ANCHOR_BLOCKS \
  $SCENE_CHANGE_TAU \
  2>&1 | tee "$OUTPUT_FOLDER/inference.log"
