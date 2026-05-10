#!/bin/bash
# wait_and_div_to_osem.sh
# 等待 PID 10160 结束，激活 conda recon 环境，
# 然后运行 div_to_osem.py

set -euo pipefail

WAIT_PID=10160
PROJECT_DIR="$HOME/graduation-thesis2"

# ── 0. 激活 conda 环境 ────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo /root/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate recon
echo "[$(date '+%H:%M:%S')] conda 环境已激活: $CONDA_DEFAULT_ENV"

# ── 1. 确认项目目录 ───────────────────────────────────────────────────────────
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: 项目目录不存在: $PROJECT_DIR" >&2
    exit 1
fi
echo "[$(date '+%H:%M:%S')] 项目目录确认: $PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/div_to_osem.py" ]; then
    echo "ERROR: 在 $PROJECT_DIR 中找不到 div_to_osem.py，请检查路径。" >&2
    exit 1
fi

# ── 2. 等待目标进程结束 ──────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] 等待 PID $WAIT_PID 结束（每 30 秒检查一次）..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] PID $WAIT_PID 已结束。"

# ── 3. 运行 div_to_osem.py ───────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] 开始运行 div_to_osem.py ..."
cd "$PROJECT_DIR"
python div_to_osem.py \
    --input_base  /root/autodl-tmp/2e9div_smooth \
    --output_dir  /root/autodl-tmp/osem_complete \
    --prefix      complete \
    --splits      train test

echo "[$(date '+%H:%M:%S')] 全部完成。"