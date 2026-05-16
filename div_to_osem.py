#!/usr/bin/env python3
"""
div_to_osem.py

Assembles divided sinogram slices into full (182, 365, 1764) sinograms,
then OSEM-reconstructs into 3D volumes (80, 128, 128).

支持三种完整数据格式（自动检测）+ 预测数据格式：
  AD      : complete_{i}_{j}.npy             (i=整数, 1-based)
  MCI     : reconstructed_{subject}_{j}.npy  (subject=字符串如 073_S_6669)
  Healthy : reconstructed_3d_image_{x}_{split}_{j}.npy
  预测     : incomplete_{i}_{j}.npy           (i=整数, 1-based)

输出 OSEM 体积统一命名为 {1-based整数}.npy，shape=(80,128,128)

用法示例：
  # 完整 AD
  python div_to_osem.py --input_base /root/autodl-tmp/2e9div_smooth_AD \
      --output_dir /root/autodl-tmp/osem/complete/AD --splits train test

  # 完整 MCI（自动检测格式，无需指定 prefix）
  python div_to_osem.py \
      --input_base /root/autodl-tmp/pet_output/2000000000/2e9div_smooth_MCI \
      --output_dir /root/autodl-tmp/osem/complete/MCI --splits train test

  # 预测（任意数据集，prefix=incomplete）
  python div_to_osem.py \
      --input_base /root/autodl-tmp/sinogram4/predicted/2e9div_smooth_MCI \
      --output_dir /root/autodl-tmp/osem/incomplete/MCI \
      --prefix incomplete --splits train test
"""

import gc
import os
import re
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sinogram_reconstruction import reconstruct_volume_from_sinogram

N_SLICES = 1764


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  格式自动检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_format(directory: str, prefix_hint: str = None) -> str:
    """
    返回文件格式标识：
      'complete_i_j'   — complete_{i}_{j}.npy
      'incomplete_i_j' — incomplete_{i}_{j}.npy
      'mci_subject_j'  — reconstructed_{subject}_{j}.npy
      'healthy_3d'     — reconstructed_3d_image_{x}_{split}_{j}.npy
    """
    if prefix_hint and prefix_hint not in ('auto', ''):
        if prefix_hint == 'complete':
            return 'complete_i_j'
        if prefix_hint == 'incomplete':
            return 'incomplete_i_j'

    files = os.listdir(directory)
    if any(f.startswith('complete_') for f in files):
        return 'complete_i_j'
    if any(f.startswith('incomplete_') for f in files):
        return 'incomplete_i_j'
    if any(f.startswith('reconstructed_3d_image_') for f in files):
        return 'healthy_3d'
    if any(f.startswith('reconstructed_') for f in files):
        return 'mci_subject_j'
    raise ValueError(f"无法识别文件格式：{directory}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  格式感知的 case_id 枚举与 sinogram 拼装
# ═══════════════════════════════════════════════════════════════════════════════

def get_cases(directory: str, fmt: str):
    """
    返回 list of (output_id: str, loader_key: any)，按 output_id 整数排序。
    output_id  : 输出文件名中的编号（1-based 整数字符串，所有格式统一）
    loader_key : assemble_sinogram 用来找文件的标识（格式相关）
    """
    if fmt in ('complete_i_j', 'incomplete_i_j'):
        pfx = 'complete' if fmt == 'complete_i_j' else 'incomplete'
        pat = re.compile(rf'^{re.escape(pfx)}_(\d+)_\d+\.npy$')
        ids = sorted({m.group(1) for f in os.listdir(directory)
                      if (m := pat.match(f))}, key=int)
        return [(cid, cid) for cid in ids]

    elif fmt == 'mci_subject_j':
        # 优先读 dataset.py 写下的 index 文件，保证与训练编号完全一致
        index_path = os.path.join(os.path.dirname(directory.rstrip('/\\')),
                                  'mci_subjects_index.txt')
        if os.path.exists(index_path):
            with open(index_path) as f:
                subjects = [l.strip() for l in f if l.strip()]
            print(f"  [MCI] 使用固定 index：{index_path}（{len(subjects)} 个患者）")
        else:
            # fallback：扫目录并按字典序排序（与 dataset.py 逻辑完全相同）
            subjects = set()
            for f in os.listdir(directory):
                if f.startswith('reconstructed_') and f.endswith('.npy'):
                    stem = f[len('reconstructed_'):-4]
                    parts = stem.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        subjects.add(parts[0])
            subjects = sorted(subjects)
            print(f"  [MCI] 警告：未找到 index 文件，按字典序排序（{len(subjects)} 个患者）")
        return [(str(idx + 1), subj) for idx, subj in enumerate(subjects)]

    elif fmt == 'healthy_3d':
        # 从文件名中探测 split（train / test）
        split = None
        for f in os.listdir(directory):
            if re.match(r'^reconstructed_3d_image_\d+_train_\d+\.npy$', f):
                split = 'train'; break
            if re.match(r'^reconstructed_3d_image_\d+_test_\d+\.npy$', f):
                split = 'test'; break
        if split is None:
            raise ValueError(f"无法从文件名判断 train/test：{directory}")

        # 优先读 index 文件
        index_path = os.path.join(os.path.dirname(directory.rstrip('/\\')),
                                  'healthy_xs_index.txt')
        if os.path.exists(index_path):
            with open(index_path) as f:
                xs = [int(l.strip()) for l in f if l.strip()]
            print(f"  [Healthy] 使用固定 index：{index_path}（{len(xs)} 个患者）")
        else:
            pat = re.compile(rf'^reconstructed_3d_image_(\d+)_{split}_\d+\.npy$')
            xs = sorted({int(m.group(1)) for f in os.listdir(directory)
                         if (m := pat.match(f))})
            print(f"  [Healthy] 警告：未找到 index 文件，按数值排序（{len(xs)} 个患者）")
        return [(str(idx + 1), (x, split)) for idx, x in enumerate(xs)]

    raise ValueError(f"未知格式：{fmt}")


def assemble_sinogram(directory: str, fmt: str, loader_key) -> np.ndarray:
    """将 1764 个切片文件拼成 (182, 365, 1764) float32。"""
    slices = []

    if fmt == 'complete_i_j':
        for j in range(1, N_SLICES + 1):
            p = os.path.join(directory, f"complete_{loader_key}_{j}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少切片：{p}")
            slices.append(np.load(p).astype(np.float32))

    elif fmt == 'incomplete_i_j':
        for j in range(1, N_SLICES + 1):
            p = os.path.join(directory, f"incomplete_{loader_key}_{j}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少切片：{p}")
            slices.append(np.load(p).astype(np.float32))

    elif fmt == 'mci_subject_j':
        subject = loader_key
        for j in range(1, N_SLICES + 1):
            p = os.path.join(directory, f"reconstructed_{subject}_{j}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少切片：{p}")
            slices.append(np.load(p).astype(np.float32))

    elif fmt == 'healthy_3d':
        x, split = loader_key
        for j in range(1, N_SLICES + 1):
            p = os.path.join(directory,
                             f"reconstructed_3d_image_{x}_{split}_{j}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少切片：{p}")
            slices.append(np.load(p).astype(np.float32))

    return np.stack(slices, axis=2)   # (182, 365, 1764)


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  OSEM 重建
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_and_save(sino: np.ndarray, out_path: str,
                         n_iters: int, n_subsets: int,
                         psf_fwhm_mm: float, use_psf: bool) -> np.ndarray:
    vol = reconstruct_volume_from_sinogram(
        sinogram_data=sino,
        n_iters=n_iters,
        n_subsets=n_subsets,
        psf_fwhm_mm=psf_fwhm_mm,
        use_psf=use_psf,
        apply_outlier_removal=False,
    )
    vol_t = vol.transpose(2, 1, 0).astype(np.float32)  # (128,128,80)→(80,128,128)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, vol_t)
    return vol_t


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  可视化
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_volume(vol: np.ndarray, case_id: str, fmt: str, out_path: str):
    D, H, W = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'OSEM [{fmt}] — case {case_id}  shape={vol.shape}', fontsize=10)
    for ax, (sl, title) in zip(axes, [
        (vol[D//2, :, :], f'Axial z={D//2}'),
        (vol[:, H//2, :], f'Coronal y={H//2}'),
        (vol[:, :, W//2], f'Sagittal x={W//2}'),
    ]):
        vmax = float(np.percentile(sl[sl > 0], 99)) if (sl > 0).any() else 1.0
        ax.imshow(sl, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Vis → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  主处理循环
# ═══════════════════════════════════════════════════════════════════════════════

def process_split(split: str, input_base: str, output_dir: str,
                  prefix_hint: str, n_iters: int, n_subsets: int,
                  psf_fwhm_mm: float, use_psf: bool, vis_dir: str,
                  only_case: str = None):

    src   = os.path.join(input_base, split)
    out   = os.path.join(output_dir, split)
    out_v = os.path.join(vis_dir,    split)
    for d in (out, out_v):
        os.makedirs(d, exist_ok=True)

    fmt = detect_format(src, prefix_hint)
    cases = get_cases(src, fmt)

    if not cases:
        print(f"  [{split}] 未找到任何文件（fmt={fmt}）：{src}"); return
    if only_case:
        cases = [(oid, lk) for oid, lk in cases if oid == only_case]
        if not cases:
            print(f"  Case {only_case} 不存在"); return

    print(f"\n[{split}] {len(cases)} cases  fmt={fmt}")

    for output_id, loader_key in tqdm(cases, desc=f"  {split}"):
        path_out = os.path.join(out,   f"{output_id}.npy")
        path_vis = os.path.join(out_v, f"{output_id}.png")
        vol = None

        if not os.path.exists(path_out):
            print(f"\n  [{output_id}] assembling + OSEM ...")
            sino = assemble_sinogram(src, fmt, loader_key)
            vol  = reconstruct_and_save(sino, path_out,
                                        n_iters, n_subsets, psf_fwhm_mm, use_psf)
            print(f"  [{output_id}] saved  shape={vol.shape}")
            del sino

        if not os.path.exists(path_vis):
            if vol is None:
                vol = np.load(path_out)
            visualize_volume(vol, output_id, fmt, path_vis)

        del vol
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Sinogram slices → OSEM → 3D volumes（自动检测 AD/MCI/Healthy/预测格式）'
    )
    parser.add_argument('--input_base',  type=str, required=True,
        help='数据根目录（含 train/ 和 test/ 子文件夹）')
    parser.add_argument('--output_dir',  type=str, required=True,
        help='输出根目录；体积保存到 output_dir/{split}/{id}.npy')
    parser.add_argument('--prefix',      type=str, default='auto',
        help='文件前缀提示："complete"、"incomplete" 或 "auto"（默认自动检测）')
    parser.add_argument('--vis_dir',     type=str, default=None,
        help='可视化输出目录（默认：output_dir + "_vis"）')
    parser.add_argument('--splits',      nargs='+', default=['train', 'test'])
    parser.add_argument('--n_iters',     type=int,   default=2)
    parser.add_argument('--n_subsets',   type=int,   default=24)
    parser.add_argument('--psf_fwhm_mm', type=float, default=4.5)
    parser.add_argument('--no_psf',      action='store_true')
    parser.add_argument('--case',        type=str,   default=None,
        help='只处理指定 case ID（调试用）')
    args = parser.parse_args()

    vis_dir = args.vis_dir or args.output_dir + '_vis'
    print(f"OSEM: iters={args.n_iters}, subsets={args.n_subsets}, "
          f"psf={'OFF' if args.no_psf else f'{args.psf_fwhm_mm}mm'}, "
          f"prefix_hint={args.prefix}")

    for split in args.splits:
        process_split(
            split=split,
            input_base=args.input_base,
            output_dir=args.output_dir,
            prefix_hint=args.prefix,
            n_iters=args.n_iters,
            n_subsets=args.n_subsets,
            psf_fwhm_mm=args.psf_fwhm_mm,
            use_psf=not args.no_psf,
            vis_dir=vis_dir,
            only_case=args.case,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
