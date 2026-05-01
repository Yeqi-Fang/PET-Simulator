"""
为新数据每个文件保存一张中间切片 PNG，方便人工筛选脑部 vs 全身扫描。
输出目录结构：
  output_dir/
    ADNI_AD/
      xxx.png
      ...
    ADNI_MCI/
      xxx.png
      ...
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

INPUT_DIRS = {
    'ADNI_AD':  r'D:\data\ADNI_AD_npy',
    'ADNI_MCI': r'D:\data\ADNI_MCI_npy',
}
OUTPUT_DIR = r'D:\data\preview_slices'

# 每个轴取中间切片，三张拼在一起方便判断
def save_preview(npy_path: Path, out_path: Path):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        print(f"  跳过 {npy_path.name}：维度为 {arr.ndim}，期望 3")
        return

    s0, s1, s2 = arr.shape
    sl0 = arr[s0 // 2, :, :]
    sl1 = arr[:, s1 // 2, :]
    sl2 = arr[:, :, s2 // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{npy_path.name}  shape={arr.shape}', fontsize=9)

    for ax, sl, label in zip(axes,
                              [sl0, sl1, sl2],
                              [f'axis0 mid={s0//2}', f'axis1 mid={s1//2}', f'axis2 mid={s2//2}']):
        vmax = np.percentile(sl, 99) or 1.0
        ax.imshow(sl, cmap='gray', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(label, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    for group, src_dir in INPUT_DIRS.items():
        src = Path(src_dir)
        if not src.exists():
            print(f"[警告] 目录不存在：{src_dir}")
            continue

        files = sorted(src.glob('*.npy'))
        print(f"\n{group}：找到 {len(files)} 个文件")

        for f in tqdm(files, desc=group):
            out = Path(OUTPUT_DIR) / group / f.with_suffix('.png').name
            save_preview(f, out)

    print(f"\n完成！预览图保存在 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
