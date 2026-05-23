#!/bin/bash
# 运行不完整环数据生成脚本

# 设置参数
NUM_EVENTS=2000000000
INPUT_DIR="/mnt/d/fyq/sinogram/reconstruction_npy_full_train/${NUM_EVENTS}/listmode"
OUTPUT_DIR="/mnt/d/fyq/sinogram/reconstruction_npy_full_train/${NUM_EVENTS}/incomplete"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行转换脚本
echo "Starting incomplete ring data generation..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

python listmode_to_incomplete.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_events "$NUM_EVENTS" \
    --visualize

echo "Conversion complete. Check $OUTPUT_DIR for results."