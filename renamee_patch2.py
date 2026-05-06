#!/usr/bin/env python3
"""
renamee_patch2.py

将遗漏的 23 个文件移动到 test/，编号 test_15 ~ test_37。

逻辑：
  - sinogram_dir 里剩余的 complete 文件，排序后依次编号 complete_15.npy ~ complete_37.npy
  - incomplete_dir 里剩余的 incomplete 文件（index194~216），排序后依次编号 incomplete_15.npy ~ incomplete_37.npy
  - 输出到 output_dir/test/

Usage:
  python renamee_patch2.py ^
      --sinogram_dir   "E:\\data\\pet_output\\2000000000\\sinogram" ^
      --incomplete_dir "D:\\data\\pet_output\\2000000000\\listmode_i_6_9_13_16_24_26_32_34\\sinogram_incomplete" ^
      --output_dir     "D:\\data\\pet_output\\2000000000\\2e9_add" ^
      --num_events     2000000000 ^
      --test_start     15
"""

import os
import glob
import shutil
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sinogram_dir',   required=True)
    parser.add_argument('--incomplete_dir', required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--num_events',     type=int, default=2000000000)
    parser.add_argument('--test_start',     type=int, default=15,
                        help='Starting local index for test set (default: 15)')
    args = parser.parse_args()

    # ── 1. 收集剩余文件 ───────────────────────────────────────────────────
    complete_files = sorted(glob.glob(
        os.path.join(args.sinogram_dir, 'reconstructed_*.npy')
    ))
    incomplete_files = sorted(glob.glob(
        os.path.join(args.incomplete_dir, f'incomplete_index*_num{args.num_events}.npy')
    ), key=lambda p: int(os.path.basename(p).split('_index')[1].split('_num')[0]))

    print(f"Remaining complete  : {len(complete_files)}")
    print(f"Remaining incomplete: {len(incomplete_files)}")

    if len(complete_files) != len(incomplete_files):
        print(f"WARNING: count mismatch! complete={len(complete_files)}, "
              f"incomplete={len(incomplete_files)}")
        print("Will process min(complete, incomplete) pairs.")

    n = min(len(complete_files), len(incomplete_files))
    if n == 0:
        print("Nothing to process.")
        return

    # ── 2. 创建输出目录 ───────────────────────────────────────────────────
    test_dir = os.path.join(args.output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output: {test_dir}")

    # ── 3. 移动并重命名 ───────────────────────────────────────────────────
    patch_csv_path = os.path.join(args.output_dir, 'patch_log.csv')
    with open(patch_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['split', 'type', 'local_idx', 'src', 'dst'])
        writer.writeheader()

        for i in range(n):
            local_idx      = args.test_start + i
            complete_src   = complete_files[i]
            incomplete_src = incomplete_files[i]
            dst_complete   = os.path.join(test_dir, f'complete_{local_idx}.npy')
            dst_incomplete = os.path.join(test_dir, f'incomplete_{local_idx}.npy')

            shutil.move(complete_src, dst_complete)
            writer.writerow({'split': 'test', 'type': 'complete',
                             'local_idx': local_idx,
                             'src': complete_src, 'dst': dst_complete})

            shutil.move(incomplete_src, dst_incomplete)
            writer.writerow({'split': 'test', 'type': 'incomplete',
                             'local_idx': local_idx,
                             'src': incomplete_src, 'dst': dst_incomplete})

            print(f"  [{i+1:2d}/{n}] test/complete_{local_idx}.npy  |  "
                  f"test/incomplete_{local_idx}.npy")

    print(f"\nDone. Moved {n} pairs → test/complete_{args.test_start} "
          f"~ test/complete_{args.test_start + n - 1}")
    print(f"Patch log → {patch_csv_path}")
    print(f"\n合并：将 {test_dir}\\*.npy 复制到 "
          f"D:\\data\\pet_output\\2000000000\\2e9\\test\\")


if __name__ == '__main__':
    main()
