#!/usr/bin/env python3
"""
renamee_patch.py

补丁脚本：读取 rename_log.csv，找出还留在 sinogram_dir / incomplete_dir
里、未被处理的文件，按正确编号移动到 --output_dir（2e9_add）。

适用场景：renamee_new.py 因 resume bug 遗漏了 N 个文件，
已处理部分不动，只补齐缺失部分。

Usage:
  python renamee_patch.py ^
      --csv            "E:\\data\\pet_output\\2000000000\\2e9\\rename_log.csv" ^
      --sinogram_dir   "E:\\data\\pet_output\\2000000000\\sinogram" ^
      --incomplete_dir "E:\\data\\pet_output\\2000000000\\listmode_i_60_120_240_300\\sinogram_incomplete" ^
      --output_dir     "D:\\data\\pet_output\\2000000000\\2e9_add" ^
      --num_events     2000000000 ^
      --n_train        180
"""

import os
import glob
import shutil
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',            required=True,
                        help='Path to rename_log.csv from the original run')
    parser.add_argument('--sinogram_dir',   required=True,
                        help='Original sinogram dir (reconstructed_*.npy)')
    parser.add_argument('--incomplete_dir', required=True,
                        help='Original incomplete dir (incomplete_index*_num*.npy)')
    parser.add_argument('--output_dir',     required=True,
                        help='Patch output root dir (2e9_add); train/ test/ created inside')
    parser.add_argument('--num_events',     type=int, default=2000000000)
    parser.add_argument('--n_train',        type=int, default=180)
    args = parser.parse_args()

    # ── 1. 读 CSV：还原已完成的 src 集合 ─────────────────────────────────────
    done_complete_srcs   = {}   # src -> dst
    done_incomplete_srcs = {}   # src -> dst

    with open(args.csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['type'] == 'complete':
                done_complete_srcs[row['src']] = row['dst']
            else:
                done_incomplete_srcs[row['src']] = row['dst']

    print(f"CSV: {len(done_complete_srcs)} complete + "
          f"{len(done_incomplete_srcs)} incomplete already done")

    # ── 2. 还原原始完整排序列表 ───────────────────────────────────────────────
    # 已移走的（来自 CSV）+ 还在 sinogram_dir 的 → 合并排序 = 原始顺序
    remaining_complete = glob.glob(
        os.path.join(args.sinogram_dir, 'reconstructed_*.npy')
    )
    all_complete = sorted(list(done_complete_srcs.keys()) + remaining_complete)
    n_total = len(all_complete)
    n_train = args.n_train
    n_test  = n_total - n_train

    print(f"Total (CSV + remaining): {n_total}  →  train={n_train}, test={n_test}")
    print(f"Files still in sinogram_dir: {len(remaining_complete)}")

    if not remaining_complete:
        print("Nothing to patch.")
        return

    # ── 3. 找出每个剩余文件对应的正确编号 ────────────────────────────────────
    # global_idx = 该文件在 all_complete 中的位置（0-based）
    remaining_set = set(remaining_complete)
    patch_items = []   # (global_idx, complete_path)
    for idx, path in enumerate(all_complete):
        if path in remaining_set:
            patch_items.append((idx, path))

    print(f"\nFiles to patch: {len(patch_items)}")
    for idx, path in patch_items:
        split     = 'train' if idx < n_train else 'test'
        local_idx = (idx + 1) if idx < n_train else (idx - n_train + 1)
        print(f"  global_idx={idx:3d}  →  {split}/complete_{local_idx}.npy"
              f"  (src: {os.path.basename(path)})")

    # ── 4. 创建输出目录 ───────────────────────────────────────────────────────
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir  = os.path.join(args.output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)
    print(f"\nOutput: {args.output_dir}")

    # ── 5. 移动 ──────────────────────────────────────────────────────────────
    patch_csv_path = os.path.join(args.output_dir, 'patch_log.csv')
    skipped = 0

    with open(patch_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['split', 'type', 'global_idx',
                                            'local_idx', 'src', 'dst'])
        writer.writeheader()
        csvfile.flush()

        for global_idx, complete_path in patch_items:
            if global_idx < n_train:
                dst_dir   = train_dir
                local_idx = global_idx + 1
                split     = 'train'
            else:
                dst_dir   = test_dir
                local_idx = global_idx - n_train + 1
                split     = 'test'

            incomplete_src = os.path.join(
                args.incomplete_dir,
                f'incomplete_index{global_idx}_num{args.num_events}.npy'
            )
            dst_complete   = os.path.join(dst_dir, f'complete_{local_idx}.npy')
            dst_incomplete = os.path.join(dst_dir, f'incomplete_{local_idx}.npy')

            # complete
            if os.path.exists(complete_path):
                shutil.move(complete_path, dst_complete)
                writer.writerow({'split': split, 'type': 'complete',
                                 'global_idx': global_idx, 'local_idx': local_idx,
                                 'src': complete_path, 'dst': dst_complete})
                csvfile.flush()
                print(f"  [C] {os.path.basename(complete_path)} → {split}/complete_{local_idx}.npy")
            else:
                print(f"  [SKIP-C] not found: {complete_path}")
                skipped += 1

            # incomplete
            if os.path.exists(incomplete_src):
                shutil.move(incomplete_src, dst_incomplete)
                writer.writerow({'split': split, 'type': 'incomplete',
                                 'global_idx': global_idx, 'local_idx': local_idx,
                                 'src': incomplete_src, 'dst': dst_incomplete})
                csvfile.flush()
                print(f"  [I] incomplete_index{global_idx} → {split}/incomplete_{local_idx}.npy")
            else:
                print(f"  [SKIP-I] incomplete not found: {incomplete_src}")
                skipped += 1

    print(f"\nPatch done. skipped={skipped}")
    print(f"Patch log → {patch_csv_path}")
    print(f"\n合并提示：")
    print(f"  将 {args.output_dir}\\train\\*.npy 复制/移动到原 2e9\\train\\ 即可")
    print(f"  将 {args.output_dir}\\test\\*.npy  复制/移动到原 2e9\\test\\  即可")


if __name__ == '__main__':
    main()
