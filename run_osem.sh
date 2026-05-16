#!/bin/bash
# run_osem.sh
# 对 AD / MCI / Healthy 三个数据集的完整和预测 sinogram 分别做 OSEM 重建
# 用法：bash run_osem.sh

set -e   # 任意命令失败立即退出

SCRIPT="python div_to_osem.py"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
AD_COMPLETE=/root/autodl-tmp/2e9div_smooth_AD
MCI_COMPLETE=/root/autodl-tmp/pet_output/2000000000/2e9div_smooth_MCI
HLT_COMPLETE=/root/autodl-tmp/2e9div_smooth_healthy

AD_PRED=/root/autodl-tmp/sinogram4/predicted/2e9div_smooth_AD
MCI_PRED=/root/autodl-tmp/sinogram4/predicted/2e9div_smooth_MCI
HLT_PRED=/root/autodl-tmp/sinogram4/predicted/2e9div_smooth_healthy

OSEM_COMPLETE=/root/autodl-tmp/osem/complete
OSEM_INCOMPLETE=/root/autodl-tmp/osem/incomplete

# ── 完整 sinogram → OSEM（自动检测格式，无需 --prefix）───────────────────────
echo "=========================================="
echo " [1/6] 完整 AD"
echo "=========================================="
$SCRIPT \
    --input_base  $AD_COMPLETE \
    --output_dir  $OSEM_COMPLETE/AD \
    --splits train test

echo "=========================================="
echo " [2/6] 完整 MCI"
echo "=========================================="
$SCRIPT \
    --input_base  $MCI_COMPLETE \
    --output_dir  $OSEM_COMPLETE/MCI \
    --splits train test

echo "=========================================="
echo " [3/6] 完整 Healthy"
echo "=========================================="
$SCRIPT \
    --input_base  $HLT_COMPLETE \
    --output_dir  $OSEM_COMPLETE/healthy \
    --splits train test

# ── 预测 sinogram → OSEM（格式为 incomplete_i_j，需指定 --prefix incomplete）─
echo "=========================================="
echo " [4/6] 预测 AD"
echo "=========================================="
$SCRIPT \
    --input_base  $AD_PRED \
    --output_dir  $OSEM_INCOMPLETE/AD \
    --prefix incomplete \
    --splits train test

echo "=========================================="
echo " [5/6] 预测 MCI"
echo "=========================================="
$SCRIPT \
    --input_base  $MCI_PRED \
    --output_dir  $OSEM_INCOMPLETE/MCI \
    --prefix incomplete \
    --splits train test

echo "=========================================="
echo " [6/6] 预测 Healthy"
echo "=========================================="
$SCRIPT \
    --input_base  $HLT_PRED \
    --output_dir  $OSEM_INCOMPLETE/healthy \
    --prefix incomplete \
    --splits train test

echo ""
echo "全部 OSEM 重建完成 ✓"
