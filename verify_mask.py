#!/usr/bin/env python3
"""
verify_mask.py

纯几何法生成 PET sinogram mask，并与实际 incomplete sinogram 对比可视化。

原理：
  对于 sinogram 中每个 (r_idx, φ_idx) 格点，
  对应的两个探测器角度为：
      d1 = φ_deg + arcsin(r_mm / R)
      d2 = φ_deg - arcsin(r_mm / R) + 180°
  若 d1 或 d2 落入缺失探测器扇区范围内，则该像素被标记为缺失（mask=0）。

  r_step = 2 * R / n_r = 2 * 253.71 / 182 ≈ 2.788 mm（体素尺寸）

Usage:
  python verify_mask.py ^
      --complete_npy   "D:\\data\\pet_output\\2000000000\\2e9\\train\\complete_1.npy" ^
      --incomplete_npy "D:\\data\\pet_output\\2000000000\\2e9\\train\\incomplete_1.npy" ^
      --slice_j        21 ^
      --output         "mask_verify.png"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False


# ── 1. 几何法 mask ────────────────────────────────────────────────────────────

def generate_mask(
    n_r: int = 182,
    n_phi: int = 365,
    R: float = 253.71,
    r_step: float = None,
    missing_sectors=None,
) -> np.ndarray:
    """
    纯几何法生成 2D mask。

    参数
    ----
    n_r     : r 轴像素数（默认 182）
    n_phi   : φ 轴像素数（默认 365）
    R       : 扫描仪半径，mm（默认 253.71）
    r_step  : 每个 r_bin 对应的 mm 数；None → 自动取 2*R/n_r ≈ 2.788 mm
    missing_sectors : 缺失探测器角度区间列表，单位度（0~360）

    返回
    ----
    mask : shape (n_r, n_phi), float32，1=有效 0=缺失
    """
    if r_step is None:
        r_step = 2.0 * R / n_r          # = 2*253.71/182 ≈ 2.788 mm

    if missing_sectors is None:
        missing_sectors = [(60, 90), (130, 160), (240, 260), (320, 340)]

    # r_mm：以 FOV 中心为零点
    r_idx = np.arange(n_r, dtype=np.float64)
    r_mm  = (r_idx - n_r / 2.0 + 0.5) * r_step          # shape (n_r,)

    # 偏移角（度）
    offset_deg = np.degrees(np.arcsin(np.clip(r_mm / R, -1.0, 1.0)))

    # φ_deg：0 ~ 180°（不含），对应 n_phi 个均匀格点
    phi_idx = np.arange(n_phi, dtype=np.float64)
    phi_deg = phi_idx * 180.0 / n_phi                    # shape (n_phi,)

    # 两个探测器角度（广播得到 n_r × n_phi 矩阵）
    d1 = offset_deg[:, np.newaxis] + phi_deg[np.newaxis, :]   # (n_r, n_phi)
    d2 = -offset_deg[:, np.newaxis] + phi_deg[np.newaxis, :] + 180.0

    def in_missing(angles):
        a = angles % 360.0
        result = np.zeros_like(a, dtype=bool)
        for s, e in missing_sectors:
            result |= (a >= s) & (a < e)
        return result

    missing_2d = in_missing(d1) | in_missing(d2)
    mask = (~missing_2d).astype(np.float32)

    print(f"  r_step = {r_step:.4f} mm  (= 2*R/n_r = 2*{R}/{n_r})")
    print(f"  max |r_mm| = {np.abs(r_mm).max():.2f} mm  "
          f"(arcsin clip = {np.degrees(np.arcsin(np.abs(r_mm).max()/R)):.1f}°)")
    return mask


# ── 2. 加载切片 ───────────────────────────────────────────────────────────────

def load_slice(npy_path: str, slice_j: int | None) -> np.ndarray:
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 3:
        if slice_j is None:
            raise ValueError("3D sinogram 需要指定 --slice_j")
        return arr[:, :, slice_j - 1]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"不支持的维度: {arr.ndim}")


# ── 3. 可视化 ─────────────────────────────────────────────────────────────────

def visualize(complete: np.ndarray,
              incomplete: np.ndarray,
              mask: np.ndarray,
              output_path: str,
              slice_j: int):

    n_r, n_phi = complete.shape
    missing_px = int((mask == 0).sum())
    total_px   = mask.size

    vmax = np.percentile(complete, 99)
    phi_bins = np.arange(n_phi)

    fig = plt.figure(figsize=(17, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.48, wspace=0.38,
                            height_ratios=[1, 1, 0.75])
    title_j = f"j={slice_j}"

    ax_c  = fig.add_subplot(gs[0, 0])   # complete slice
    ax_i  = fig.add_subplot(gs[0, 1])   # incomplete slice
    ax_m  = fig.add_subplot(gs[0, 2])   # geometric mask

    ax_mi = fig.add_subplot(gs[1, 0])   # incomplete × mask
    ax_d  = fig.add_subplot(gs[1, 1])   # difference
    ax_ov = fig.add_subplot(gs[1, 2])   # incomplete + red highlight

    ax_p  = fig.add_subplot(gs[2, :])   # φ profile

    # ── Row 0 ───────────────────────────────────────────────────────────────
    im = ax_c.imshow(complete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax_c.set_title(f'Complete sinogram  ({title_j})')
    ax_c.set_xlabel('φ bin'); ax_c.set_ylabel('r bin')
    plt.colorbar(im, ax=ax_c, fraction=0.04)

    ax_i.imshow(incomplete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax_i.set_title(f'Incomplete sinogram  ({title_j})')
    ax_i.set_xlabel('φ bin'); ax_i.set_ylabel('r bin')

    ax_m.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax_m.set_title('几何 mask（r_step=2R/n_r=2.788 mm）\n绿=有效  红=缺失')
    ax_m.set_xlabel('φ bin'); ax_m.set_ylabel('r bin')
    ax_m.text(0.02, 0.97,
              f'缺失: {missing_px}/{total_px}\n({100*missing_px/total_px:.1f}%)',
              transform=ax_m.transAxes, fontsize=8, va='top',
              bbox=dict(fc='white', alpha=0.7))

    # ── Row 1 ───────────────────────────────────────────────────────────────
    masked_incomplete = incomplete * mask
    ax_mi.imshow(masked_incomplete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax_mi.set_title(f'Incomplete × mask  ({title_j})')
    ax_mi.set_xlabel('φ bin'); ax_mi.set_ylabel('r bin')

    diff = incomplete * (1 - mask)           # 缺失区的实际计数（应全为 0）
    nonzero = int((diff != 0).sum())
    vd = max(diff.max(), 1e-6)
    ax_d.imshow(diff, cmap='bwr', vmin=-vd, vmax=vd, aspect='auto')
    ax_d.set_title(f'Incomplete × (1−mask)  缺失区计数\n非零像素: {nonzero}  (期望 0)')
    ax_d.set_xlabel('φ bin'); ax_d.set_ylabel('r bin')

    # Overlay: incomplete + red where mask=0
    ax_ov.imshow(incomplete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    overlay = np.zeros((*incomplete.shape, 4), dtype=np.float32)
    overlay[mask == 0, 0] = 1.0
    overlay[mask == 0, 3] = 0.45
    ax_ov.imshow(overlay, aspect='auto')
    ax_ov.set_title(f'Incomplete + mask=0 高亮（红色）\n{title_j}')
    ax_ov.set_xlabel('φ bin'); ax_ov.set_ylabel('r bin')

    # ── Row 2: φ-profile ────────────────────────────────────────────────────
    profile_c = complete.mean(axis=0)
    profile_i = incomplete.mean(axis=0)
    missing_center = np.where(mask[n_r // 2] == 0)[0]

    ax_p.plot(phi_bins, profile_c, color='steelblue', lw=1.2, label='Complete (mean over r)')
    ax_p.plot(phi_bins, profile_i, color='orange',    lw=1.2, label='Incomplete (mean over r)')
    for k in missing_center:
        ax_p.axvspan(k - 0.5, k + 0.5, color='red', alpha=0.15)
    ax_p.axvspan(-1, -1, color='red', alpha=0.25, label='Mask=0（中心行）')
    ax_p.set_xlim(0, n_phi - 1)
    ax_p.set_xlabel('φ bin'); ax_p.set_ylabel('Mean counts')
    ax_p.set_title('φ 轴平均计数剖线')
    ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.3)

    fig.suptitle('Sinogram Mask 验证（几何法，r_step=2R/n_r≈2.788 mm）',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"图像已保存: {output_path}")

    # ── 统计 ─────────────────────────────────────────────────────────────────
    print(f"\n── Mask 统计 ──────────────────────────────────────────")
    print(f"  总像素  : {total_px}")
    print(f"  缺失像素: {missing_px}  ({100*missing_px/total_px:.1f}%)")
    print(f"  有效像素: {total_px-missing_px}  ({100*(total_px-missing_px)/total_px:.1f}%)")
    print(f"\n── 一致性验证 ─────────────────────────────────────────")
    print(f"  缺失区域 incomplete 非零像素: {nonzero}  (期望 0)")
    if mask.sum() > 0:
        ratio = incomplete[mask == 1].sum() / (complete[mask == 1].sum() + 1e-9)
        print(f"  有效区域 incomplete/complete 比: {ratio:.4f}  (期望 ~0.5)")

    # 保存 mask
    mask_path = output_path.replace('.png', '_mask.npy')
    np.save(mask_path, mask)
    print(f"  Mask 已保存: {mask_path}")


# ── 4. main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='几何法生成并验证 PET sinogram mask')
    parser.add_argument('--complete_npy',   type=str, required=True,
                        help='complete sinogram .npy（2D 或 3D）')
    parser.add_argument('--incomplete_npy', type=str, required=True,
                        help='incomplete sinogram .npy（2D 或 3D）')
    parser.add_argument('--slice_j', type=int, default=21,
                        help='3D 时的切片编号（1-based，默认 21）')
    parser.add_argument('--r_step', type=float, default=None,
                        help='r_bin 对应 mm 数（默认 2*R/n_r ≈ 2.788）')
    parser.add_argument('--output', type=str, default='mask_verify.png',
                        help='输出图像路径（同时保存 _mask.npy）')
    args = parser.parse_args()

    print(f"加载 complete:   {args.complete_npy}")
    complete_3d = np.load(args.complete_npy).astype(np.float32)
    print(f"加载 incomplete: {args.incomplete_npy}")
    incomplete_3d = np.load(args.incomplete_npy).astype(np.float32)
    print(f"Shape: {complete_3d.shape}")

    assert complete_3d.shape == incomplete_3d.shape, \
        f"shape 不一致: {complete_3d.shape} vs {incomplete_3d.shape}"

    complete   = load_slice(args.complete_npy,   args.slice_j) if complete_3d.ndim == 3 \
                 else complete_3d
    incomplete = load_slice(args.incomplete_npy, args.slice_j) if incomplete_3d.ndim == 3 \
                 else incomplete_3d

    # 重新加载（load_slice 内部也会 load，这里统一）
    complete   = load_slice(args.complete_npy,   args.slice_j)
    incomplete = load_slice(args.incomplete_npy, args.slice_j)

    n_r, n_phi = complete.shape
    print(f"切片 shape: ({n_r}, {n_phi})")

    print(f"\n正在生成几何 mask（r_step={'auto' if args.r_step is None else args.r_step} mm）...")
    mask = generate_mask(n_r=n_r, n_phi=n_phi, r_step=args.r_step)

    visualize(complete, incomplete, mask, args.output, args.slice_j)


if __name__ == '__main__':
    main()
