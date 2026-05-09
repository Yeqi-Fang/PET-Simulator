#!/usr/bin/env python3
"""
verify_pairs.py

データペアの整合性を検証する診断スクリプト。
complete_{i}.npy と incomplete_{i}.npy が同一被験者由来かどうかを確認する。

原理：
  正しいペアなら incomplete = complete * mask のはず。
  つまり incomplete が 0 の領域では complete も相対的に小さく、
  incomplete が正の領域では complete も正でなければならない。

  簡単な検証：
    Pearsonの相関係数が高い → 同一被験者
    相関係数が低い → 別被験者（ペア不一致）

Usage:
  python verify_pairs.py --data_dir "C:\\path\\to\\2e9div" --split train --n_check 10
"""

import os
import argparse
import numpy as np
import glob


def check_pair(complete_path, incomplete_path):
    """
    complete と incomplete が同一被験者由来かチェックする。

    Returns:
        dict with:
            corr: Pearson相関係数（1に近いほど同一被験者らしい）
            ratio: incomplete/complete の比（~0.5のはず）
            is_valid: 有効なペアかどうか（True/False）
    """
    try:
        comp = np.load(complete_path).astype(np.float32).ravel()
        inc  = np.load(incomplete_path).astype(np.float32).ravel()
    except Exception as e:
        return {'corr': None, 'ratio': None, 'is_valid': False, 'error': str(e)}

    if comp.shape != inc.shape:
        return {'corr': None, 'ratio': None, 'is_valid': False,
                'error': f'shape mismatch: {comp.shape} vs {inc.shape}'}

    # 全体のPearson相関係数（形状の類似度）
    comp_c = comp - comp.mean()
    inc_c  = inc  - inc.mean()
    denom  = np.sqrt((comp_c**2).sum() * (inc_c**2).sum()) + 1e-12
    corr   = float((comp_c * inc_c).sum() / denom)

    # incomplete/complete 比（有効領域のみ）
    valid_mask = comp > (comp.max() * 0.01)  # complete の 1% 以上
    if valid_mask.sum() > 0:
        ratio = float(inc[valid_mask].sum() / comp[valid_mask].sum())
    else:
        ratio = None

    # 判定：同一被験者なら相関が高いはず（経験的しきい値 0.9）
    is_valid = corr > 0.85

    return {'corr': corr, 'ratio': ratio, 'is_valid': is_valid}


def main():
    parser = argparse.ArgumentParser(description='Verify complete/incomplete sinogram pairing')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base data dir (contains train/ and test/)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Which split to check')
    parser.add_argument('--n_check', type=int, default=10,
                        help='How many subject indices to check (default: 10)')
    parser.add_argument('--start_i', type=int, default=1,
                        help='Starting subject index (default: 1)')
    parser.add_argument('--j', type=int, default=42,
                        help='Which angle slice to check (default: 42, 1-based)')
    parser.add_argument('--check_all_i', action='store_true',
                        help='Check all available subjects (ignores n_check)')
    args = parser.parse_args()

    split_dir = os.path.join(args.data_dir, args.split)
    if not os.path.isdir(split_dir):
        print(f"ERROR: directory not found: {split_dir}")
        return

    # --- 利用可能な i インデックスを探す ---
    complete_files = sorted(glob.glob(os.path.join(split_dir, f"complete_*_{args.j}.npy")))
    if not complete_files:
        # j なしの場合（complete_i.npy 形式）
        complete_files = sorted(glob.glob(os.path.join(split_dir, "complete_*.npy")))
        use_j = False
    else:
        use_j = True

    if not complete_files:
        print(f"ERROR: No complete_*.npy files found in {split_dir}")
        return

    print(f"Found {len(complete_files)} complete files in {split_dir}")
    print(f"Checking j={args.j} slice\n")

    # 確認する i の範囲
    if args.check_all_i:
        if use_j:
            indices = []
            for fp in complete_files:
                base = os.path.basename(fp)  # complete_3_42.npy
                parts = base.replace('.npy', '').split('_')
                try:
                    indices.append(int(parts[1]))
                except:
                    pass
        else:
            indices = []
            for fp in complete_files:
                base = os.path.basename(fp)  # complete_3.npy
                parts = base.replace('.npy', '').split('_')
                try:
                    indices.append(int(parts[1]))
                except:
                    pass
    else:
        indices = list(range(args.start_i, args.start_i + args.n_check))

    # --- 各ペアを検証 ---
    bad_pairs = []
    good_pairs = []

    for i in sorted(set(indices)):
        if use_j:
            comp_path = os.path.join(split_dir, f"complete_{i}_{args.j}.npy")
            inc_path  = os.path.join(split_dir, f"incomplete_{i}_{args.j}.npy")
        else:
            comp_path = os.path.join(split_dir, f"complete_{i}.npy")
            inc_path  = os.path.join(split_dir, f"incomplete_{i}.npy")

        if not os.path.exists(comp_path):
            print(f"  [i={i}] SKIP: complete file not found: {comp_path}")
            continue
        if not os.path.exists(inc_path):
            print(f"  [i={i}] SKIP: incomplete file not found: {inc_path}")
            continue

        result = check_pair(comp_path, inc_path)

        status = "OK   " if result['is_valid'] else "BAD  "
        corr_str  = f"{result['corr']:.4f}" if result['corr'] is not None else "N/A"
        ratio_str = f"{result['ratio']:.4f}" if result['ratio'] is not None else "N/A"

        print(f"  [{status}] i={i:3d}  corr={corr_str}  inc/comp_ratio={ratio_str}")

        if result['is_valid']:
            good_pairs.append(i)
        else:
            bad_pairs.append(i)

    print(f"\n{'='*60}")
    print(f"Summary for split={args.split}, j={args.j}:")
    print(f"  Good pairs: {len(good_pairs)}")
    print(f"  Bad  pairs: {len(bad_pairs)}")
    if bad_pairs:
        print(f"  BAD indices: {bad_pairs}")
        print(f"\n=> ペア不一致が検出されました！")
        print(f"   complete_{{i}}.npy と incomplete_{{i}}.npy が別被験者のデータです。")
        print(f"   renamee_patch2.py のペアリングバグが原因の可能性が高いです。")
    else:
        print(f"  => 全ペア正常（相関 > 0.85）")


if __name__ == '__main__':
    main()
