#!/usr/bin/env python3
"""
smooth_and_split.py

将 2e9/train 和 2e9/test 中的 complete/incomplete .npy 文件
逐一做 smooth → split(float16) → 删除 smooth 中间文件，
避免同时在磁盘上存储完整的 smooth 数据集。

磁盘用量峰值：
  D：2e9（~194 GB，保留）+ 单个 smooth 临时文件（~0.5 GB）
  C：2e9div_smooth（~47 GB，持续增长）

断点续跑：检测目标目录中第 1764 个切片是否已存在，存在则跳过该文件。

Usage（示例）：
  python smooth_and_split.py ^
      --input_base   "D:\\data\\pet_output\\2000000000\\2e9" ^
      --output_base  "C:\\data\\pet_output\\2000000000\\2e9div_smooth" ^
      --tmp_dir      "D:\\data\\pet_output\\2000000000\\smooth_tmp"
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pytomography.io.PET import gate

# ── PET scanner info ────────────────────────────────────────────────────────
INFO = {
    'min_rsector_difference': np.float32(0.0),
    'crystal_length':          np.float32(0.0),
    'radius':                  np.float32(253.71),
    'crystalTransNr':          13,
    'crystalTransSpacing':     np.float32(4.01648),
    'crystalAxialNr':          7,
    'crystalAxialSpacing':     np.float32(5.36556),
    'submoduleAxialNr':        1,
    'submoduleAxialSpacing':   np.float32(0.0),
    'submoduleTransNr':        1,
    'submoduleTransSpacing':   np.float32(0.0),
    'moduleTransNr':           1,
    'moduleTransSpacing':      np.float32(0.0),
    'moduleAxialNr':           6,
    'moduleAxialSpacing':      np.float32(37.55892),
    'rsectorTransNr':          28,
    'rsectorAxialNr':          1,
    'TOF':                     0,
    'NrCrystalsPerRing':       364,
    'NrRings':                 42,
    'firstCrystalAxis':        0,
}


def smooth_array(data: np.ndarray) -> np.ndarray:
    """Apply gate.smooth_randoms_sinogram and return numpy array (float32)."""
    tensor = torch.tensor(data)
    smoothed = gate.smooth_randoms_sinogram(
        tensor, INFO,
        sigma_r=0.7, sigma_theta=0.7, sigma_z=0.7,
        kernel_size_r=5, kernel_size_theta=5, kernel_size_z=5,
    )
    return smoothed.cpu().numpy()


def split_and_save(data: np.ndarray, base_name: str, out_dir: str) -> None:
    """
    Split axis-2 of data (shape H×W×1764) into 1764 float16 files.
    Saves as {base_name}_{j}.npy  (j = 1-based).
    """
    os.makedirs(out_dir, exist_ok=True)
    for j in range(data.shape[2]):
        slice_fp = os.path.join(out_dir, f"{base_name}_{j + 1}.npy")
        np.save(slice_fp, data[:, :, j].astype(np.float16))


def already_done(base_name: str, out_dir: str, n_slices: int = 1764) -> bool:
    """Return True if the last slice already exists (resume check)."""
    last_slice = os.path.join(out_dir, f"{base_name}_{n_slices}.npy")
    return os.path.exists(last_slice)


def process_file(src_path: str, tmp_dir: str, out_dir: str) -> None:
    """
    Load → smooth (tmp file on D) → split to out_dir (on C) → delete tmp.
    """
    base_name = os.path.splitext(os.path.basename(src_path))[0]  # e.g. complete_1

    # ── 断点续跑：最后一片已存在则跳过 ────────────────────────────────────
    if already_done(base_name, out_dir):
        return  # silently skip

    # ── 1. Load ─────────────────────────────────────────────────────────────
    data = np.load(src_path)
    if not (data.ndim == 3 and data.shape[2] == 1764):
        print(f"  [SKIP] unexpected shape {data.shape}: {src_path}")
        return

    # ── 2. Smooth → write temp file ─────────────────────────────────────────
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"_smooth_tmp_{base_name}.npy")
    smoothed = smooth_array(data)
    del data  # free memory
    np.save(tmp_path, smoothed)

    # ── 3. Split → write float16 slices to C ────────────────────────────────
    split_and_save(smoothed, base_name, out_dir)
    del smoothed  # free memory

    # ── 4. Delete temp file ─────────────────────────────────────────────────
    try:
        os.remove(tmp_path)
    except OSError as e:
        print(f"  [WARN] Could not delete tmp file {tmp_path}: {e}")


def process_split(split_name: str, input_base: str, output_base: str, tmp_dir: str) -> None:
    """Process one split (train or test)."""
    src_dir = os.path.join(input_base, split_name)
    out_dir = os.path.join(output_base, split_name)

    npy_files = sorted([
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith('.npy')
    ])

    print(f"\n[{split_name}] {len(npy_files)} files  →  {out_dir}")

    skipped = 0
    for fp in tqdm(npy_files, desc=split_name):
        base_name = os.path.splitext(os.path.basename(fp))[0]
        if already_done(base_name, out_dir):
            skipped += 1
            continue
        process_file(fp, tmp_dir, out_dir)

    print(f"[{split_name}] done. skipped(already done)={skipped}")


def main():
    parser = argparse.ArgumentParser(description='Smooth + split sinograms, one file at a time')
    parser.add_argument('--input_base',  required=True,
                        help='Root dir with train/ and test/ subdirs (e.g. D:\\...\\2e9)')
    parser.add_argument('--output_base', required=True,
                        help='Root output dir for split float16 files (e.g. C:\\...\\2e9div_smooth)')
    parser.add_argument('--tmp_dir',     required=True,
                        help='Temp dir for smooth intermediates (should be on same drive as input_base)')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        help='Subdirectory names to process (default: train test)')
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.input_base, args.output_base, args.tmp_dir)

    print("\nAll done.")
    # Clean up tmp_dir if empty
    try:
        os.rmdir(args.tmp_dir)
        print(f"Removed empty tmp dir: {args.tmp_dir}")
    except OSError:
        pass  # not empty or doesn't exist — fine


if __name__ == '__main__':
    main()
