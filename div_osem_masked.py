#!/usr/bin/env python3
"""
div_osem_masked.py

在 div_to_osem.py 基础上新增 --apply-mask 选项：
将几何 LOR mask（来自 verify_mask.py）应用到完整 sinogram，
模拟不完整探测器环后再做 OSEM 重建。

mask 来源（按优先级）：
  1. --mask-npy <path>                    显式指定 .npy 文件
  2. 同目录下的 sinogram_mask_cache.npy   自动查找缓存
  3. verify_mask.generate_mask(INFO)      实时生成

用法：
  # 不完整 AD（mask sinogram → OSEM）
  python div_osem_masked.py \\
      --input_base /root/autodl-tmp/2e9div_smooth_AD \\
      --output_dir /root/autodl-tmp/osem_direct/incomplete/AD \\
      --apply-mask \\
      --splits train test

  # 指定 mask 文件
  python div_osem_masked.py \\
      --input_base /root/autodl-tmp/2e9div_smooth_AD \\
      --output_dir /root/autodl-tmp/osem_direct/incomplete/AD \\
      --apply-mask --mask-npy sinogram_mask_cache.npy \\
      --splits train test
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
from verify_mask import generate_mask, INFO as SCANNER_INFO

N_SLICES = 1764


# ═══════════════════════════════════════════════════════════════════════════════
#  0.  Mask 加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_mask(mask_npy: str = None) -> np.ndarray:
    """
    返回 (n_angle, n_radial) float32 mask（1=有效, 0=缺失）。
    """
    # 1. 显式路径
    if mask_npy and os.path.isfile(mask_npy):
        m = np.load(mask_npy).astype(np.float32)
        print(f"Mask 加载自：{mask_npy}  shape={m.shape}")
        return m

    # 2. 同目录缓存
    cache = os.path.join(_HERE, 'sinogram_mask_cache.npy')
    if os.path.isfile(cache):
        m = np.load(cache).astype(np.float32)
        print(f"Mask 加载自缓存：{cache}  shape={m.shape}")
        return m

    # 3. 实时生成并缓存
    print("生成 sinogram mask（首次运行）...")
    m, _, _ = generate_mask(SCANNER_INFO)
    m = m.astype(np.float32)
    np.save(cache, m)
    print(f"Mask 已缓存至：{cache}  shape={m.shape}")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  格式自动检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_format(directory: str, prefix_hint: str = None) -> str:
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
    if fmt in ('complete_i_j', 'incomplete_i_j'):
        pfx = 'complete' if fmt == 'complete_i_j' else 'incomplete'
        pat = re.compile(rf'^{re.escape(pfx)}_(\d+)_\d+\.npy$')
        ids = sorted({m.group(1) for f in os.listdir(directory)
                      if (m := pat.match(f))}, key=int)
        return [(cid, cid) for cid in ids]

    elif fmt == 'mci_subject_j':
        index_path = os.path.join(os.path.dirname(directory.rstrip('/\\')),
                                  'mci_subjects_index.txt')
        if os.path.exists(index_path):
            with open(index_path) as f:
                subjects = [l.strip() for l in f if l.strip()]
            print(f"  [MCI] 使用固定 index：{index_path}（{len(subjects)} 个患者）")
        else:
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
        split = None
        for f in os.listdir(directory):
            if re.match(r'^reconstructed_3d_image_\d+_train_\d+\.npy$', f):
                split = 'train'; break
            if re.match(r'^reconstructed_3d_image_\d+_test_\d+\.npy$', f):
                split = 'test'; break
        if split is None:
            raise ValueError(f"无法从文件名判断 train/test：{directory}")

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
    fig.suptitle(f'OSEM masked [{fmt}] — case {case_id}  shape={vol.shape}', fontsize=10)
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
                  mask: np.ndarray = None,
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

    print(f"\n[{split}] {len(cases)} cases  fmt={fmt}"
          + ("  [masked]" if mask is not None else ""))

    for output_id, loader_key in tqdm(cases, desc=f"  {split}"):
        path_out = os.path.join(out,   f"{output_id}.npy")
        path_vis = os.path.join(out_v, f"{output_id}.png")
        vol = None

        if not os.path.exists(path_out):
            print(f"\n  [{output_id}] assembling + mask + OSEM ...")
            sino = assemble_sinogram(src, fmt, loader_key)   # (182, 365, 1764)

            if mask is not None:
                sino *= mask[:, :, np.newaxis]   # broadcast over axial axis

            vol = reconstruct_and_save(sino, path_out,
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
        description='Sinogram → mask → OSEM → 3D volumes'
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
    parser.add_argument('--apply-mask',  action='store_true',
        help='应用几何 LOR mask，模拟不完整探测器环（默认开启）')
    parser.add_argument('--mask-npy',    type=str, default=None,
        help='mask .npy 路径；不指定则自动从缓存或 verify_mask.py 生成')

    args = parser.parse_args()

    vis_dir = args.vis_dir or args.output_dir + '_vis'

    # 加载 mask
    mask = load_mask(mask_npy=args.mask_npy)
    if not args.apply_mask:
        print("注意：未指定 --apply-mask，mask 不会应用（等同于 div_to_osem.py）")
        mask = None
    else:
        n_zero = int((mask == 0).sum())
        print(f"Mask: {n_zero}/{mask.size} bins zeroed ({100*n_zero/mask.size:.1f}%)")

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
            mask=mask,
            only_case=args.case,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
