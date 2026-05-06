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
      --sinogram_dir   "E:\\data\\pet_output\\2000000000\\sinogram" ^
      --incomplete_dir "E:\\data\\pet_output\\2000000000\\listmode_i_60_120_240_300\\sinogram_incomplete" ^
      --output_dir     "E:\\data\\pet_output\\2000000000\\2e9" ^
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

    csv_path = os.path.join(args.output_dir, 'rename_log.csv')

    # ── 1. 从 CSV 还原已完成的 src→dst 映射 ──────────────────────────────────
    # BUG FIX: 不能在 resume 时重新 glob sinogram_dir 来构建 complete_files，
    # 因为已移走的文件不在那里了，导致 global_idx 错位。
    # 正确做法：用 CSV 里记录的原始 src 路径 + sinogram_dir 中剩余文件，
    # 合并排序还原出原始有序列表。
    done_complete_src2dst = {}   # src_path -> dst_path
    done_incomplete_src2dst = {} # src_path -> dst_path

    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['type'] == 'complete':
                    done_complete_src2dst[row['src']] = row['dst']
                else:
                    done_incomplete_src2dst[row['src']] = row['dst']
        print(f"Resume mode: {len(done_complete_src2dst)} complete + "
              f"{len(done_incomplete_src2dst)} incomplete already done in CSV")

    # ── 2. 还原原始 complete 文件列表（已移走的 + 仍在目录里的，统一排序）──
    remaining_complete = glob.glob(os.path.join(args.sinogram_dir, 'reconstructed_*.npy'))
    complete_files = sorted(
        list(done_complete_src2dst.keys()) + remaining_complete
    )
    n_total = len(complete_files)
    print(f"Total complete sinograms (CSV+remaining): {n_total}")

    if n_total == 0:
        print("ERROR: no complete sinogram files found")
        return

    n_train = args.n_train
    n_test  = n_total - n_train
    print(f"Split: train={n_train}, test={n_test}")

    # ── 3. 验证 incomplete 数量 ───────────────────────────────────────────────
    remaining_incomplete = glob.glob(
        os.path.join(args.incomplete_dir, f'incomplete_index*_num{args.num_events}.npy')
    )
    n_incomplete = len(done_incomplete_src2dst) + len(remaining_incomplete)
    print(f"Total incomplete sinograms (CSV+remaining): {n_incomplete}")
    if n_incomplete != n_total:
        print(f"WARNING: count mismatch ({n_incomplete} incomplete vs {n_total} complete). "
              "Missing files will be skipped.")

    # ── 4. 创建输出目录 ───────────────────────────────────────────────────────
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir  = os.path.join(args.output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)
    print(f"Output dirs:\n  train → {train_dir}\n  test  → {test_dir}")

    # ── 5. 移动并重命名 ───────────────────────────────────────────────────────
    skipped  = 0
    resumed  = 0
    is_new   = (len(done_complete_src2dst) == 0 and len(done_incomplete_src2dst) == 0)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['split', 'type', 'src', 'dst'])
        if is_new:
            writer.writeheader()
            csvfile.flush()

        for global_idx, complete_path in enumerate(complete_files):
            j = global_idx
            incomplete_path = os.path.join(
                args.incomplete_dir,
                f'incomplete_index{j}_num{args.num_events}.npy'
            )

            if global_idx < n_train:
                dst_dir   = train_dir
                local_idx = global_idx + 1
                split     = 'train'
            else:
                dst_dir   = test_dir
                local_idx = global_idx - n_train + 1
                split     = 'test'

            dst_complete   = os.path.join(dst_dir, f'complete_{local_idx}.npy')
            dst_incomplete = os.path.join(dst_dir, f'incomplete_{local_idx}.npy')

            # ── 移动 complete ────────────────────────────────────────────────
            if complete_path in done_complete_src2dst:
                # 已在 CSV 里记录：验证 dst 一致且文件存在
                recorded_dst = done_complete_src2dst[complete_path]
                if recorded_dst == dst_complete and os.path.exists(dst_complete):
                    resumed += 1  # 正确完成，跳过
                else:
                    print(f"  [ERROR] Mismatch at global_idx={global_idx}: "
                          f"CSV dst={recorded_dst}, expected={dst_complete}, "
                          f"exists={os.path.exists(dst_complete)}")
            elif os.path.exists(dst_complete):
                # dst 已存在但不在 CSV → 残缺文件，删除重移
                print(f"  [WARN] Removing stale dst: {dst_complete}")
                os.remove(dst_complete)
                shutil.move(complete_path, dst_complete)
                writer.writerow({'split': split, 'type': 'complete',
                                 'src': complete_path, 'dst': dst_complete})
                csvfile.flush()
            elif os.path.exists(complete_path):
                shutil.move(complete_path, dst_complete)
                writer.writerow({'split': split, 'type': 'complete',
                                 'src': complete_path, 'dst': dst_complete})
                csvfile.flush()
            else:
                print(f"  [ERROR] src not found and not in CSV: {complete_path}")

            # ── 移动 incomplete ──────────────────────────────────────────────
            if incomplete_path in done_incomplete_src2dst:
                recorded_dst = done_incomplete_src2dst[incomplete_path]
                if recorded_dst == dst_incomplete and os.path.exists(dst_incomplete):
                    pass  # 正确完成，跳过
                else:
                    print(f"  [WARN] Incomplete mismatch at global_idx={global_idx}")
            elif os.path.exists(dst_incomplete):
                print(f"  [WARN] Removing stale dst: {dst_incomplete}")
                os.remove(dst_incomplete)
                if os.path.exists(incomplete_path):
                    shutil.move(incomplete_path, dst_incomplete)
                    writer.writerow({'split': split, 'type': 'incomplete',
                                     'src': incomplete_path, 'dst': dst_incomplete})
                    csvfile.flush()
            elif os.path.exists(incomplete_path):
                shutil.move(incomplete_path, dst_incomplete)
                writer.writerow({'split': split, 'type': 'incomplete',
                                 'src': incomplete_path, 'dst': dst_incomplete})
                csvfile.flush()
            else:
                print(f"  [WARN] incomplete not found, skipping: {incomplete_path}")
                skipped += 1

            if (global_idx + 1) % 20 == 0 or global_idx + 1 == n_total:
                print(f"  [{global_idx+1}/{n_total}] done (resumed={resumed}, skipped={skipped})")

    print(f"\nFinished. skipped={skipped}")
    print(f"  train: complete_1~{n_train}.npy  /  incomplete_1~{n_train}.npy")
    print(f"  test:  complete_1~{n_test}.npy   /  incomplete_1~{n_test}.npy")
    print(f"  Rename log → {csv_path}")


if __name__ == '__main__':
    main()
