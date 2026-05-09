#!/usr/bin/env python3
"""
div_to_osem.py

Assembles divided sinogram slices ({prefix}_{N}_{j}.npy, j=1..1764)
into full (182, 365, 1764) sinograms, then OSEM-reconstructs into 3D volumes.

Output shape: (80, 128, 128) per volume  [= OSEM (128,128,80).transpose(2,1,0)]

Examples:
  # complete sinograms
  python div_to_osem.py \
      --input_base /root/autodl-tmp/2e9div_smooth \
      --output_dir /root/autodl-tmp/osem_complete \
      --prefix complete --splits test

  # sinogram4-predicted sinograms
  python div_to_osem.py \
      --input_base /root/autodl-tmp/prediction \
      --output_dir /root/autodl-tmp/osem_predicted \
      --prefix incomplete --splits test
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
#  1.  Sinogram assembly
# ═══════════════════════════════════════════════════════════════════════════════

def get_case_ids(directory: str, prefix: str):
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)_\d+\.npy$')
    ids = {m.group(1) for f in os.listdir(directory)
           if (m := pattern.match(f))}
    return sorted(ids, key=lambda x: int(x))


def assemble_sinogram(directory: str, prefix: str, case_id: str) -> np.ndarray:
    """Stack {prefix}_{case_id}_{j}.npy slices → (182, 365, 1764) float32."""
    slices = []
    for j in range(1, N_SLICES + 1):
        path = os.path.join(directory, f"{prefix}_{case_id}_{j}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing slice: {path}")
        slices.append(np.load(path).astype(np.float32))
    return np.stack(slices, axis=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  OSEM reconstruction wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_and_save(sino: np.ndarray, out_path: str,
                         n_iters: int, n_subsets: int,
                         psf_fwhm_mm: float, use_psf: bool) -> np.ndarray:
    """OSEM → transpose(2,1,0) → save float32. Returns (80,128,128) volume."""
    vol = reconstruct_volume_from_sinogram(
        sinogram_data=sino,
        n_iters=n_iters,
        n_subsets=n_subsets,
        psf_fwhm_mm=psf_fwhm_mm,
        use_psf=use_psf,
        apply_outlier_removal=False,
    )
    # OSEM returns (128,128,80); AttentionUNet3D needs (80,128,128)
    vol_t = vol.transpose(2, 1, 0).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, vol_t)
    return vol_t


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_volume(vol: np.ndarray, case_id: str, prefix: str, out_path: str):
    """3 orthogonal slices of the OSEM volume."""
    D, H, W = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'OSEM [{prefix}] — case {case_id}  shape={vol.shape}', fontsize=10)

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
    print(f"  Visualization → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Main processing loop
# ═══════════════════════════════════════════════════════════════════════════════

def process_split(split: str,
                  input_base: str,
                  output_dir: str,
                  prefix: str,
                  n_iters: int, n_subsets: int,
                  psf_fwhm_mm: float, use_psf: bool,
                  vis_dir: str,
                  only_case: str = None):

    src   = os.path.join(input_base, split)
    out   = os.path.join(output_dir, split)
    out_v = os.path.join(vis_dir,    split)
    for d in (out, out_v):
        os.makedirs(d, exist_ok=True)

    case_ids = get_case_ids(src, prefix)
    if not case_ids:
        print(f"  No {prefix}_* files found in {src}"); return
    if only_case:
        case_ids = [c for c in case_ids if c == only_case]
        if not case_ids:
            print(f"  Case {only_case} not found"); return

    print(f"\n[{split}] {len(case_ids)} cases  prefix={prefix}")

    for case_id in tqdm(case_ids, desc=f"  {split}"):
        path_out = os.path.join(out,   f"{case_id}.npy")
        path_vis = os.path.join(out_v, f"{case_id}.png")

        vol = None

        if not os.path.exists(path_out):
            print(f"\n  [{case_id}] assembling + OSEM ...")
            sino = assemble_sinogram(src, prefix, case_id)
            vol  = reconstruct_and_save(sino, path_out,
                                        n_iters, n_subsets, psf_fwhm_mm, use_psf)
            print(f"  [{case_id}] saved  shape={vol.shape}")
            del sino

        if not os.path.exists(path_vis):
            if vol is None:
                vol = np.load(path_out)
            visualize_volume(vol, case_id, prefix, path_vis)

        del vol
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Assemble divided sinogram slices → OSEM → 3D volumes'
    )
    parser.add_argument('--input_base',  type=str, required=True,
        help='Base dir with train/test subfolders containing {prefix}_{i}_{j}.npy')
    parser.add_argument('--output_dir',  type=str, required=True,
        help='Output base dir; volumes go to output_dir/{split}/{i}.npy')
    parser.add_argument('--prefix',      type=str, default='complete',
        help='Filename prefix: "complete" or "incomplete" (default: complete)')
    parser.add_argument('--vis_dir',     type=str, default=None,
        help='Visualization output dir (default: output_dir_vis)')
    parser.add_argument('--splits',      nargs='+', default=['train', 'test'])
    parser.add_argument('--n_iters',     type=int,   default=2)
    parser.add_argument('--n_subsets',   type=int,   default=24)
    parser.add_argument('--psf_fwhm_mm', type=float, default=4.5)
    parser.add_argument('--no_psf',      action='store_true')
    parser.add_argument('--case',        type=str,   default=None,
        help='Process only one case ID (for debugging)')
    args = parser.parse_args()

    vis_dir = args.vis_dir or args.output_dir + '_vis'

    print(f"OSEM: iters={args.n_iters}, subsets={args.n_subsets}, "
          f"psf={'OFF' if args.no_psf else f'{args.psf_fwhm_mm}mm'}, "
          f"prefix={args.prefix}")

    for split in args.splits:
        process_split(
            split=split,
            input_base=args.input_base,
            output_dir=args.output_dir,
            prefix=args.prefix,
            n_iters=args.n_iters, n_subsets=args.n_subsets,
            psf_fwhm_mm=args.psf_fwhm_mm, use_psf=not args.no_psf,
            vis_dir=vis_dir,
            only_case=args.case,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
