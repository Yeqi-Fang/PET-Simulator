"""
可视化新旧数据各三个轴的切片，帮助判断轴向对应关系。

旧数据：D:\PET\dataset\train_npy_crop  (已知 80×128×128，axis0=轴向)
新数据：D:\data\ADNI_AD_npy 或 D:\data\ADNI_MCI_npy  (160×160×96，轴向待确认)

输出 6 张图：
  old_axis0.png  old_axis1.png  old_axis2.png
  new_axis0.png  new_axis1.png  new_axis2.png
每张图显示沿该轴均匀采样的 5 个切片。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OLD_DIR = r'D:\PET\dataset\train_npy_crop'
NEW_DIRS = [r'D:\data\ADNI_AD_npy_rotated', r'D:\data\ADNI_MCI_npy_rotated']
OUTPUT_DIR = r'D:\data\compare_axes'
N_SLICES = 5   # 每张图显示多少个切片


def pick_sample(directory: str) -> np.ndarray:
    """从目录中随机取第一个有效 .npy 文件"""
    d = Path(directory)
    files = sorted(d.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f"找不到 .npy 文件：{directory}")
    arr = np.load(files[0])
    print(f"  采样文件：{files[0].name}  shape={arr.shape}")
    return arr


def save_axis_figure(arr: np.ndarray, axis: int, label: str, out_path: Path):
    """沿指定轴均匀取 N_SLICES 张切片并保存为 PNG"""
    n = arr.shape[axis]
    indices = np.linspace(0, n - 1, N_SLICES, dtype=int)

    fig, axes = plt.subplots(1, N_SLICES, figsize=(N_SLICES * 3, 3.5))
    fig.suptitle(f'{label}  —  沿 axis{axis} 切片  (shape={arr.shape})', fontsize=10)

    for ax, idx in zip(axes, indices):
        sl = np.take(arr, idx, axis=axis)
        vmax = np.percentile(sl, 99) or 1.0
        ax.imshow(sl, cmap='gray', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(f'idx={idx}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  已保存：{out_path}")


def main():
    out = Path(OUTPUT_DIR)

    # --- 旧数据 ---
    print(f"\n[旧数据] {OLD_DIR}")
    try:
        old_arr = pick_sample(OLD_DIR)
        for axis in range(3):
            save_axis_figure(old_arr, axis, '旧数据 (80×128×128, axis0=轴向)',
                             out / f'old_axis{axis}.png')
    except FileNotFoundError as e:
        print(f"  [跳过] {e}")

    # --- 新数据 ---
    new_arr = None
    for d in NEW_DIRS:
        print(f"\n[新数据] {d}")
        try:
            new_arr = pick_sample(d)
            break
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")

    if new_arr is not None:
        for axis in range(3):
            save_axis_figure(new_arr, axis, '新数据 (160×160×96, 轴向待确认)',
                             out / f'new_axis{axis}.png')

    print(f"\n完成！6 张对比图保存在 {OUTPUT_DIR}")
    print("对照方式：找旧数据中 axis0（冠状/横断面）对应新数据哪个轴，即可确认轴向。")


if __name__ == '__main__':
    main()
