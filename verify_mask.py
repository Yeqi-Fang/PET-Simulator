#!/usr/bin/env python3
"""
verify_mask.py

离散探测器对法生成 PET sinogram mask，并与实际 incomplete sinogram 对比可视化。

原理：
  枚举单个 ring 内所有 transaxial crystal pair，
  将其中"任意一端落在缺失扇区"的 bad LOR 用 gate.listmode_to_sinogram
  映射到 sinogram bin，得到 bad_sino；
  mask = 0 → bad_sino > 0 的 bin（缺失 LOR 命中的格点）
  mask = 1 → 其余有统计支撑的格点

  sinogram shape: (n_angle=182, n_radial=365, n_axial=1764)
  axis 0 = angle, axis 1 = radial（不是 r×phi！）

  detector 物理角度 = (crystal_index × 360 / N + 90°) % 360°
  （simulator 中 effective_angles = linspace(0,2π,N) + π/2，
    所以 crystal 0 的物理位置在 +y 方向 = 90°）

Usage:
  python verify_mask.py ^
      --complete_npy   "D:\\data\\pet_output\\2000000000\\2e9\\train\\complete_1.npy" ^
      --incomplete_npy "D:\\data\\pet_output\\2000000000\\2e9\\train\\incomplete_1.npy" ^
      --slice_j        21 ^
      --output         "mask_verify.png"
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from pytomography.io.PET import gate

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False


# ── Scanner INFO ──────────────────────────────────────────────────────────────

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

# 缺失探测器扇区（crystal index 角度，°）
# angle_deg = (360 / NrCrystalsPerRing) * crystal_idx，crystal 0 → 0°
MISSING_SECTORS = [(60, 90), (130, 160), (240, 260), (320, 340)]


# ── 1. 判断每个 crystal 是否缺失 ─────────────────────────────────────────────

def get_missing_crystals(info: dict,
                         missing_sectors=None) -> np.ndarray:
    """
    返回 shape (NrCrystalsPerRing,) 的布尔数组，True = 该 crystal 缺失。

    角度公式与 listmode_to_incomplete.py 完全一致：
        angle_deg = (360 / N) * crystal_idx   （crystal 0 = 0°，无任何偏移）
    """
    if missing_sectors is None:
        missing_sectors = MISSING_SECTORS

    N = int(info['NrCrystalsPerRing'])
    idx = np.arange(N, dtype=np.float64)
    angle_deg = idx * 360.0 / N              # 与 listmode_to_incomplete.py 一致

    missing = np.zeros(N, dtype=bool)
    for s, e in missing_sectors:
        if s <= e:
            missing |= (angle_deg >= s) & (angle_deg <= e)
        else:                                 # 跨越 0°/360° 边界
            missing |= (angle_deg >= s) | (angle_deg <= e)

    n_miss = missing.sum()
    print(f"  缺失 crystal: {n_miss}/{N}  ({100*n_miss/N:.1f}%)")
    idxs = np.where(missing)[0]
    print(f"  缺失 crystal index 范围: [{idxs[0]}, {idxs[-1]}]  "
          f"(角度 {idxs[0]*360.0/N:.1f}° ~ {idxs[-1]*360.0/N:.1f}°)")
    return missing


# ── 2. 用 gate.listmode_to_sinogram 生成 mask ─────────────────────────────────

def generate_mask(info: dict,
                  missing_sectors=None,
                  ring_idx: int = 20) -> tuple:
    """
    枚举 ring_idx 内所有 transaxial detector pair，
    用 listmode_to_sinogram 映射到 sinogram bin，生成 2D mask。

    返回
    ----
    mask    : shape (n_angle, n_radial), float32，1=有效 0=缺失
    support : shape (n_angle, n_radial), bool，True=有物理 LOR 的 bin
    z_idx   : 使用的 axial slice 索引（用于信息打印）
    """
    N = int(info['NrCrystalsPerRing'])
    missing_crystal = get_missing_crystals(info, missing_sectors)

    # ── 枚举同一 ring 内所有无序 crystal pair (i < j) ─────────────────────────
    ci = np.arange(N)
    ii, jj = np.meshgrid(ci, ci, indexing='ij')
    upper = ii < jj
    i_arr = ii[upper]
    j_arr = jj[upper]

    # flat detector ID = ring_idx * N + within_ring_id
    det1 = (ring_idx * N + i_arr).astype(np.int32)
    det2 = (ring_idx * N + j_arr).astype(np.int32)

    all_pairs = torch.from_numpy(np.stack([det1, det2], axis=1))  # (M, 2)
    bad_mask  = missing_crystal[i_arr] | missing_crystal[j_arr]
    bad_pairs = all_pairs[bad_mask]

    print(f"  总 LOR 对: {len(all_pairs):,}   缺失 LOR 对: {bad_mask.sum():,}")

    # ── 调用 listmode_to_sinogram ──────────────────────────────────────────────
    with torch.no_grad():
        sino_all = gate.listmode_to_sinogram(all_pairs, info).numpy()  # (182, 365, 1764)
        sino_bad = gate.listmode_to_sinogram(bad_pairs, info).numpy()

    print(f"  sinogram shape: {sino_all.shape}  "
          f"(axis0=angle, axis1=radial, axis2=axial)")

    # ── 找该 ring 对应的 axial slice（计数最多的那个） ─────────────────────────
    axial_counts = sino_all.sum(axis=(0, 1))
    z_idx = int(axial_counts.argmax())
    print(f"  axial slice 索引 (ring {ring_idx}×{ring_idx}): {z_idx}")

    sl_all = sino_all[:, :, z_idx]   # (n_angle, n_radial)
    sl_bad = sino_bad[:, :, z_idx]

    support = sl_all > 0
    bad_bins = sl_bad > 0

    mask_2d = np.ones(sl_all.shape, dtype=np.float32)
    mask_2d[bad_bins] = 0.0

    miss_pct = 100 * bad_bins.sum() / support.sum() if support.sum() > 0 else 0
    print(f"  support 内缺失 bin: {bad_bins[support].sum()} / {support.sum()}"
          f"  ({miss_pct:.1f}%)")

    return mask_2d, support, z_idx


# ── 3. 加载切片 ───────────────────────────────────────────────────────────────

def load_slice(npy_path: str, slice_j: int) -> np.ndarray:
    """
    加载 .npy 文件并返回 2D 切片，shape = (n_angle, n_radial)。
    支持 2D（直接返回）或 3D（沿 axis=2 取第 slice_j-1 片）。
    """
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 3:
        return arr[:, :, slice_j - 1]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"不支持的维度: {arr.ndim}")


# ── 4. 可视化 ─────────────────────────────────────────────────────────────────

def make_empirical_mask(complete_3d: np.ndarray,
                        incomplete_3d: np.ndarray,
                        threshold_ratio: float = 0.02) -> np.ndarray:
    """
    经验法：对所有 axial slice 求和后，
    complete_sum > threshold 且 incomplete_sum == 0 → 真实缺失。
    返回 shape (n_angle, n_radial), 1=有效 0=缺失
    """
    c_sum = complete_3d.sum(axis=2)
    i_sum = incomplete_3d.sum(axis=2)
    thr   = c_sum.max() * threshold_ratio
    missing = (c_sum > thr) & (i_sum == 0)
    return (~missing).astype(np.float32)


def visualize(complete: np.ndarray,
              incomplete: np.ndarray,
              mask_geo: np.ndarray,
              mask_emp: np.ndarray,
              support: np.ndarray,
              output_path: str,
              slice_j: int):

    n_angle, n_radial = complete.shape
    support_px  = int(support.sum())
    total_px    = mask_geo.size

    def miss_pct(m):
        n = int((m[support] == 0).sum())
        return n, 100 * n / max(support_px, 1)

    geo_miss, geo_pct = miss_pct(mask_geo)
    emp_miss, emp_pct = miss_pct(mask_emp)

    vmax = np.percentile(complete[complete > 0], 99) if (complete > 0).any() else 1.0

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            height_ratios=[1, 1, 0.75])
    tj = f"j={slice_j}"

    ax_c   = fig.add_subplot(gs[0, 0])
    ax_i   = fig.add_subplot(gs[0, 1])
    ax_geo = fig.add_subplot(gs[0, 2])
    ax_emp = fig.add_subplot(gs[1, 0])
    ax_fp  = fig.add_subplot(gs[1, 1])
    ax_ov  = fig.add_subplot(gs[1, 2])
    ax_p   = fig.add_subplot(gs[2, :])

    # ── Row 0 ────────────────────────────────────────────────────────────────
    im = ax_c.imshow(complete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax_c.set_title(f'Complete sinogram  ({tj})')
    ax_c.set_xlabel('Radial bin'); ax_c.set_ylabel('Angle bin')
    plt.colorbar(im, ax=ax_c, fraction=0.04)

    ax_i.imshow(incomplete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax_i.set_title(f'Incomplete sinogram  ({tj})')
    ax_i.set_xlabel('Radial bin'); ax_i.set_ylabel('Angle bin')

    ax_geo.imshow(mask_geo, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax_geo.set_title(f'几何 mask（离散探测器对法）\n缺失: {geo_miss} ({geo_pct:.1f}%)')
    ax_geo.set_xlabel('Radial bin'); ax_geo.set_ylabel('Angle bin')

    # ── Row 1 ────────────────────────────────────────────────────────────────
    ax_emp.imshow(mask_emp, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax_emp.set_title(f'经验 mask（complete/incomplete 求和推导）\n缺失: {emp_miss} ({emp_pct:.1f}%)')
    ax_emp.set_xlabel('Radial bin'); ax_emp.set_ylabel('Angle bin')

    # 误差图：几何 mask 和经验 mask 的差异
    # +1 = 几何漏标（几何=1 但经验=0，即几何认为有效但经验认为缺失）
    # -1 = 几何误杀（几何=0 但经验=1，即几何认为缺失但经验认为有效）
    err_map = mask_geo.astype(np.float32) - mask_emp.astype(np.float32)
    false_kill  = int((err_map[support] < 0).sum())   # 几何=0, 经验=1
    false_alive = int((err_map[support] > 0).sum())   # 几何=1, 经验=0
    im_e = ax_fp.imshow(err_map, cmap='bwr', vmin=-1, vmax=1, aspect='auto')
    ax_fp.set_title(f'几何mask − 经验mask\n'
                    f'蓝=几何误杀({false_kill})  红=几何漏标({false_alive})')
    ax_fp.set_xlabel('Radial bin'); ax_fp.set_ylabel('Angle bin')
    plt.colorbar(im_e, ax=ax_fp, fraction=0.04)

    # Overlay: incomplete + 几何 mask=0 红色高亮
    ax_ov.imshow(incomplete, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    overlay = np.zeros((*incomplete.shape, 4), dtype=np.float32)
    overlay[mask_geo == 0, 0] = 1.0
    overlay[mask_geo == 0, 3] = 0.40
    ax_ov.imshow(overlay, aspect='auto')
    ax_ov.set_title(f'Incomplete + 几何mask=0 高亮（红）  ({tj})')
    ax_ov.set_xlabel('Radial bin'); ax_ov.set_ylabel('Angle bin')

    # ── Row 2: 两种 mask 的 angle-averaged radial 剖线对比 ───────────────────
    prof_c = complete.mean(axis=0)
    prof_i = incomplete.mean(axis=0)
    prof_geo = mask_geo.mean(axis=0)   # 按 angle 平均后的有效比例
    prof_emp = mask_emp.mean(axis=0)

    radial_bins = np.arange(n_radial)
    ax_p.plot(radial_bins, prof_c / (prof_c.max() + 1e-9),
              color='steelblue', lw=1.2, label='Complete (归一化)')
    ax_p.plot(radial_bins, prof_i / (prof_c.max() + 1e-9),
              color='orange', lw=1.2, label='Incomplete (归一化)')
    ax_p.plot(radial_bins, prof_geo,
              color='green', lw=1.2, ls='--', label='几何mask（mean over angle）')
    ax_p.plot(radial_bins, prof_emp,
              color='red', lw=1.2, ls=':', label='经验mask（mean over angle）')
    ax_p.set_xlim(0, n_radial - 1)
    ax_p.set_xlabel('Radial bin'); ax_p.set_ylabel('Counts / Mask 有效比')
    ax_p.set_title('Radial 剖线：sinogram 计数 + 两种 mask 有效比对比')
    ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

    fig.suptitle('Sinogram Mask 验证：几何法 vs 经验法', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图像已保存: {output_path}")

    # ── 统计 ─────────────────────────────────────────────────────────────────
    print(f"\n── Mask 统计（support 内）──────────────────────────")
    print(f"  几何 mask 缺失: {geo_miss} ({geo_pct:.1f}%)")
    print(f"  经验 mask 缺失: {emp_miss} ({emp_pct:.1f}%)")
    print(f"  几何误杀 (geo=0, emp=1): {false_kill}")
    print(f"  几何漏标 (geo=1, emp=0): {false_alive}")
    iou_missing = int(((mask_geo==0) & (mask_emp==0) & support).sum())
    union       = int(((mask_geo==0) | (mask_emp==0) & support).sum())
    print(f"  缺失区域 IoU: {iou_missing}/{union} = {iou_missing/(union+1e-9):.3f}")

    valid_mask = mask_geo == 1
    if valid_mask.any():
        ratio = incomplete[valid_mask].sum() / (complete[valid_mask].sum() + 1e-9)
        print(f"  几何有效区 incomplete/complete 比: {ratio:.4f}  (期望 ~0.5)")

    # 保存几何 mask
    mask_path = output_path.replace('.png', '_mask.npy')
    np.save(mask_path, mask_geo)
    print(f"\n  几何 Mask 已保存: {mask_path}")


# ── 5. main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='离散探测器对法生成并验证 PET sinogram mask')
    parser.add_argument('--complete_npy',   required=True,
                        help='complete sinogram .npy（2D 或 3D）')
    parser.add_argument('--incomplete_npy', required=True,
                        help='incomplete sinogram .npy（2D 或 3D）')
    parser.add_argument('--slice_j', type=int, default=21,
                        help='3D 时的切片编号（1-based，默认 21）')
    parser.add_argument('--ring_idx', type=int, default=20,
                        help='用于生成 mask 的 ring 编号（0-based，默认 20）')
    parser.add_argument('--output', type=str, default='mask_verify.png',
                        help='输出图像路径（同时保存 _mask.npy）')
    args = parser.parse_args()

    print(f"加载 complete:   {args.complete_npy}")
    print(f"加载 incomplete: {args.incomplete_npy}")
    complete   = load_slice(args.complete_npy,   args.slice_j)
    incomplete = load_slice(args.incomplete_npy, args.slice_j)
    print(f"切片 shape: {complete.shape}  (axis0=angle, axis1=radial)")

    assert complete.shape == incomplete.shape, \
        f"shape 不一致: {complete.shape} vs {incomplete.shape}"

    print(f"\n正在生成 mask（ring_idx={args.ring_idx}）...")
    mask, support, z_idx = generate_mask(INFO, ring_idx=args.ring_idx)

    assert mask.shape == complete.shape, \
        (f"mask shape {mask.shape} 与 sinogram shape {complete.shape} 不匹配，"
         f"请检查 INFO 中的 NrCrystalsPerRing 是否为 {complete.shape[1]-1}")

    visualize(complete, incomplete, mask, support, args.output, args.slice_j)


if __name__ == '__main__':
    main()
