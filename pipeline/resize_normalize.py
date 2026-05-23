"""
将旋转后的新数据 (96, 160, 160) 处理成与旧数据一致的格式 (80, 128, 128)。

轴约定（axis0=轴向Axial, axis1=冠状Coronal, axis2=矢状Sagittal）：
  axis0 (96 → 77 → pad → 80)：轴向，横断面
  axis1 (160 → 128)：冠状
  axis2 (160 → 128)：矢状

处理步骤：
  1. 归一化：除以体素最大值 → [0, 1]
  2. 等比缩放 0.8：(96,160,160) → (77,128,128)
  3. axis0 居中填零：77 → 80
  4. 保存为 float64
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize

TASKS = [
    (r'D:\data\ADNI_AD_npy_rotated',  r'D:\data\ADNI_AD_npy_processed'),
    (r'D:\data\ADNI_MCI_npy_rotated', r'D:\data\ADNI_MCI_npy_processed'),
]

INPUT_SHAPE  = (96, 160, 160)
RESIZE_SHAPE = (77, 128, 128)   # 96*0.8=76.8 → 77
TARGET_SHAPE = (80, 128, 128)   # 与旧数据一致


def process(arr: np.ndarray) -> np.ndarray:
    # 1. 归一化到 [0, 1]
    vmax = arr.max()
    if vmax > 0:
        arr = arr / vmax

    # 2. 等比缩放到 (77, 128, 128)，双线性插值
    arr_resized = resize(
        arr,
        RESIZE_SHAPE,
        order=1,              # 双线性
        mode='constant',
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
    )

    # 3. axis0 居中零填充：77 → 80，前补2后补1
    pad_total  = TARGET_SHAPE[0] - RESIZE_SHAPE[0]   # = 3
    pad_before = pad_total // 2                        # = 1
    pad_after  = pad_total - pad_before                # = 2
    arr_padded = np.pad(
        arr_resized,
        ((pad_before, pad_after), (0, 0), (0, 0)),
        mode='constant',
        constant_values=0,
    )

    return arr_padded.astype(np.float64)


def main():
    for src_dir, dst_dir in TASKS:
        src = Path(src_dir)
        dst = Path(dst_dir)

        if not src.exists():
            print(f"[跳过] 目录不存在：{src_dir}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob('*.npy'))
        print(f"\n{src.name}：{len(files)} 个文件 → {dst_dir}")

        for f in tqdm(files, desc=src.name):
            arr = np.load(f)

            if arr.shape != INPUT_SHAPE:
                tqdm.write(f"  [跳过] {f.name}：shape={arr.shape}，期望 {INPUT_SHAPE}")
                continue

            out = process(arr)

            assert out.shape == TARGET_SHAPE, f"输出 shape 异常：{out.shape}"
            assert out.dtype == np.float64

            np.save(dst / f.name, out)

        print(f"  完成，输出 shape={TARGET_SHAPE}，dtype=float64，值域=[0,1]")


if __name__ == '__main__':
    main()
