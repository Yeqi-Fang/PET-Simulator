"""
将新 PET 数据 (160, 160, 96) 转换为与旧数据相同的方向 (96, 160, 160)。

推导：
  old axis0 (128×128)  ↔  new axis2 (160×160)，old CW 90° = new
  old axis1 (80×128)   ↔  new axis1 (160×96)，  old CW 90° + flip = new
  old axis2 (80×128)   ↔  new axis0 (160×96)，  old CCW 90° = new

一致推导得：T[z, y, x] = N[x, 159-y, z]
即：np.flip(N, axis=1).transpose(2, 1, 0)

如果保存后对比发现 axis2（矢状面）左右镜像，把 FLIP_AXIS2 改为 True。
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

TASKS = [
    (r'D:\data\ADNI_AD_npy',  r'D:\data\ADNI_AD_npy_rotated'),
    (r'D:\data\ADNI_MCI_npy', r'D:\data\ADNI_MCI_npy_rotated'),
]

EXPECTED_SHAPE = (160, 160, 96)

# 如果变换后 axis2（矢状面）左右镜像，改为 True
FLIP_AXIS2 = False


def transform(arr: np.ndarray) -> np.ndarray:
    """
    (160, 160, 96) → (96, 160, 160)

    步骤：
      1. flip axis1：N[a, b, c] → N[a, 159-b, c]
      2. transpose(2,1,0)：轴顺序从 (a,b,c) 变为 (c,b,a)
    结果：T[z, y, x] = N[x, 159-y, z]
    """
    result = np.flip(arr, axis=1).transpose(2, 1, 0)
    if FLIP_AXIS2:
        result = np.flip(result, axis=2)
    return np.ascontiguousarray(result.astype(arr.dtype))


def main():
    for src_dir, dst_dir in TASKS:
        src = Path(src_dir)
        dst = Path(dst_dir)

        if not src.exists():
            print(f"[跳过] 目录不存在：{src_dir}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob('*.npy'))
        print(f"\n{src.name}：共 {len(files)} 个文件 → {dst_dir}")

        skipped = 0
        for f in tqdm(files, desc=src.name):
            arr = np.load(f)
            if arr.shape != EXPECTED_SHAPE:
                tqdm.write(f"  [跳过] {f.name}：shape={arr.shape}，期望 {EXPECTED_SHAPE}")
                skipped += 1
                continue
            out = transform(arr)
            np.save(dst / f.name, out)

        print(f"  完成，跳过 {skipped} 个文件（shape 不符）")
        if skipped:
            print(f"  被跳过的文件可能是全身扫描或其他 shape，请用 preview_new_data.py 检查后手动处理")

    print("\n全部完成！")
    print("建议用 compare_axes.py 对比 rotated 目录与旧数据，确认方向一致。")
    print("如发现 axis2（矢状面）仍然镜像，将脚本顶部的 FLIP_AXIS2 改为 True 重跑。")


if __name__ == '__main__':
    main()
