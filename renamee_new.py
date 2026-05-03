#!/usr/bin/env python3
"""
renamee_new.py

将 complete sinogram 和 incomplete sinogram 按排序顺序重命名，
并分配到 train / test 目录。

对应关系：
  complete sinogram:   {sinogram_dir}/reconstructed_{image_filename}.npy
                       按文件名字母序排序，第 i 个（1-based）→ complete_{i}.npy
  incomplete sinogram: {incomplete_sinogram_dir}/incomplete_index{j}_num{num_events}.npy
                       j 是 0-based，与 complete 一一对应 → incomplete_{j+1}.npy

划分方式（plan A, 共 217 个）：
  前 180 个 → train/
  后  37 个 → test/
  两个目录内的编号都从 1 开始重新计数：
    train: complete_1.npy … complete_180.npy  /  incomplete_1.npy … incomplete_180.npy
    test:  complete_1.npy … complete_37.npy   /  incomplete_1.npy … incomplete_37.npy

Usage:
  python renamee_new.py ^
      --sinogram_dir        "E:\\data\\pet_output\\2000000000\\sinogram" ^
      --incomplete_dir      "E:\\data\\pet_output\\2000000000\\listmode_i_60_120_240_300\\sinogram_incomplete" ^
      --output_dir          "E:\\data\\pet_output\\2000000000\\2e9" ^
      --num_events          2000000000 ^
      --n_train             180
"""

import os
import glob
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sinogram_dir', type=str, required=True,
                        help='Dir with reconstructed_{image_filename}.npy (complete sinograms)')
    parser.add_argument('--incomplete_dir', type=str, required=True,
                        help='Dir with incomplete_index{j}_num{num_events}.npy')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root output dir; train/ and test/ will be created inside')
    parser.add_argument('--num_events', type=int, default=2000000000)
    parser.add_argument('--n_train', type=int, default=180,
                        help='Number of files for training set (rest go to test)')
    args = parser.parse_args()

    # ── 1. 收集并排序 complete sinogram ────────────────────────────────────
    complete_files = sorted(glob.glob(os.path.join(args.sinogram_dir, 'reconstructed_*.npy')))
    n_total = len(complete_files)
    print(f"Found {n_total} complete sinograms in {args.sinogram_dir}")

    if n_total == 0:
        print("ERROR: no complete sinogram files found, check --sinogram_dir")
        return

    n_train = args.n_train
    n_test  = n_total - n_train
    print(f"Split: train={n_train}, test={n_test}")

    # ── 2. 验证 incomplete sinogram 数量 ────────────────────────────────────
    incomplete_files_check = glob.glob(
        os.path.join(args.incomplete_dir, f'incomplete_index*_num{args.num_events}.npy')
    )
    print(f"Found {len(incomplete_files_check)} incomplete sinograms in {args.incomplete_dir}")
    if len(incomplete_files_check) != n_total:
        print(f"WARNING: count mismatch ({len(incomplete_files_check)} incomplete vs {n_total} complete). "
              "Continuing anyway — missing files will be skipped.")

    # ── 3. 创建输出目录 ───────────────────────────────────────────────────────
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir  = os.path.join(args.output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)
    print(f"Output dirs:\n  train → {train_dir}\n  test  → {test_dir}")

    # ── 4. 复制并重命名 ───────────────────────────────────────────────────────
    skipped = 0
    for global_idx, complete_path in enumerate(complete_files):
        # 0-based global index → 0-based j for incomplete
        j = global_idx                          # incomplete_index{j}_num{num_events}.npy
        incomplete_path = os.path.join(
            args.incomplete_dir,
            f'incomplete_index{j}_num{args.num_events}.npy'
        )

        # 决定属于 train 还是 test，以及目录内 1-based 编号
        if global_idx < n_train:
            dst_dir    = train_dir
            local_idx  = global_idx + 1          # 1-based within train
        else:
            dst_dir    = test_dir
            local_idx  = global_idx - n_train + 1  # 1-based within test

        # 目标路径
        dst_complete   = os.path.join(dst_dir, f'complete_{local_idx}.npy')
        dst_incomplete = os.path.join(dst_dir, f'incomplete_{local_idx}.npy')

        # 复制 complete
        shutil.copy2(complete_path, dst_complete)

        # 复制 incomplete（文件可能不存在）
        if os.path.exists(incomplete_path):
            shutil.copy2(incomplete_path, dst_incomplete)
        else:
            print(f"  [WARN] incomplete not found, skipping: {incomplete_path}")
            skipped += 1

        if (global_idx + 1) % 20 == 0 or global_idx + 1 == n_total:
            print(f"  [{global_idx+1}/{n_total}] done")

    print(f"\nFinished. skipped={skipped}")
    print(f"  train: complete_1~{n_train}.npy  /  incomplete_1~{n_train}.npy")
    print(f"  test:  complete_1~{n_test}.npy   /  incomplete_1~{n_test}.npy")


if __name__ == '__main__':
    main()
