#!/usr/bin/env python3
"""
compute_missing_coincidence.py

给定一组 sinogram 路径和缺失扇区定义，
计算每个 sinogram 因探测器缺失而损失的符合事件比例（missing coincidence level）。

missing coincidence level = 落在 mask=0 bin 中的计数总和 / sinogram 总计数
例如：0.50 表示约 50% 的事件因探测器缺失而丢失。

所有 mask 生成逻辑直接从 verify_mask.py 导入，不重复实现。

Usage:
  # 使用默认缺失扇区（verify_mask.py 中的 MISSING_SECTORS）
  python compute_missing_coincidence.py ^
      --sinograms path/to/sino1.npy path/to/sino2.npy

  # 自定义缺失扇区（格式: "start1,end1;start2,end2;..."，角度单位°）
  python compute_missing_coincidence.py ^
      --sinograms path/to/sino1.npy ^
      --missing_sectors "60,90;130,160;240,260;320,340"

  # 指定 ring_idx
  python compute_missing_coincidence.py ^
      --sinograms path/to/sino1.npy ^
      --missing_sectors "60,90;240,260" ^
      --ring_idx 20
"""

import argparse
import numpy as np

# ── 直接从 verify_mask.py 导入，不重写任何 mask 生成代码 ──────────────────────
from verify_mask import generate_mask, INFO, MISSING_SECTORS


# ── 辅助：解析命令行扇区字符串 ────────────────────────────────────────────────

def parse_sectors(sector_str: str) -> list:
    """
    解析缺失扇区字符串。
    格式: "start1,end1;start2,end2;..."  （角度，°）
    示例: "60,90;130,160;240,260;320,340"
    返回: [(60.0, 90.0), (130.0, 160.0), ...]
    """
    sectors = []
    for pair in sector_str.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(',')
        if len(parts) != 2:
            raise ValueError(f"扇区格式错误: '{pair}'，期望 'start,end'")
        sectors.append((float(parts[0].strip()), float(parts[1].strip())))
    return sectors


# ── 核心：计算单个 sinogram 的 missing coincidence level ──────────────────────

def compute_missing_level(sino_path: str, mask_2d: np.ndarray) -> dict:
    """
    加载 sinogram，将 2D mask 广播到全部 axial slices，
    统计落在 mask=0（缺失）bin 内的计数占总计数的比例。

    参数
    ----
    sino_path : sinogram .npy 文件路径（支持 2D 或 3D）
    mask_2d   : shape (n_angle, n_radial)，1=有效 0=缺失，由 generate_mask 生成

    返回
    ----
    dict 包含:
      path           : 文件路径
      shape          : 数组 shape
      total_counts   : sinogram 总计数
      missing_counts : 落在缺失 bin 的计数
      missing_level  : missing_counts / total_counts（小数形式）
    """
    arr = np.load(sino_path).astype(np.float64)

    if arr.ndim == 2:
        if arr.shape != mask_2d.shape:
            raise ValueError(
                f"2D sinogram shape {arr.shape} 与 mask shape {mask_2d.shape} 不匹配"
            )
        total   = arr.sum()
        missing = arr[mask_2d == 0].sum()

    elif arr.ndim == 3:
        n_angle, n_radial, n_axial = arr.shape
        if (n_angle, n_radial) != mask_2d.shape:
            raise ValueError(
                f"sinogram transaxial shape ({n_angle},{n_radial}) "
                f"与 mask shape {mask_2d.shape} 不匹配"
            )
        # 将 2D mask 广播到所有 axial slice：同一 transaxial 模式对每层均适用
        mask_3d = np.broadcast_to(
            mask_2d[:, :, np.newaxis], (n_angle, n_radial, n_axial)
        )
        total   = arr.sum()
        missing = arr[mask_3d == 0].sum()

    else:
        raise ValueError(f"不支持的 sinogram 维度: {arr.ndim}，期望 2D 或 3D")

    level = float(missing / total) if total > 0 else 0.0

    return {
        'path':           sino_path,
        'shape':          arr.shape,
        'total_counts':   total,
        'missing_counts': missing,
        'missing_level':  level,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='计算 PET sinogram 因探测器缺失损失的符合事件比例',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--sinograms', nargs='+', required=True,
        metavar='PATH',
        help='一个或多个 sinogram .npy 文件路径',
    )
    parser.add_argument(
        '--missing_sectors', type=str, default=None,
        metavar='SECTORS',
        help=(
            '缺失扇区（角度，°），格式 "start1,end1;start2,end2;..."\n'
            '不指定则使用 verify_mask.py 中的默认值 MISSING_SECTORS'
        ),
    )
    parser.add_argument(
        '--ring_idx', type=int, default=20,
        help='用于生成 mask 的 ring 编号（0-based，默认 20）',
    )
    args = parser.parse_args()

    # ── 解析缺失扇区 ──────────────────────────────────────────────────────────
    if args.missing_sectors is not None:
        sectors = parse_sectors(args.missing_sectors)
        print(f"自定义缺失扇区: {sectors}")
    else:
        sectors = MISSING_SECTORS
        print(f"使用默认缺失扇区 (from verify_mask.MISSING_SECTORS): {sectors}")

    # ── 生成 2D mask（调用 verify_mask.generate_mask，不重复实现） ─────────────
    print(f"\n正在生成 mask（ring_idx={args.ring_idx}）...")
    mask_2d, support, z_idx = generate_mask(INFO, missing_sectors=sectors,
                                            ring_idx=args.ring_idx)
    print(f"  2D mask shape: {mask_2d.shape}  "
          f"有效 bin: {int((mask_2d == 1).sum())}  "
          f"缺失 bin: {int((mask_2d == 0).sum())}")

    # ── 逐个 sinogram 计算 missing coincidence level ──────────────────────────
    print(f"\n{'='*70}")
    print(f"{'文件':<46} {'Missing Level':>13}  {'Missing/Total Counts'}")
    print(f"{'='*70}")

    results = []
    for path in args.sinograms:
        try:
            r = compute_missing_level(path, mask_2d)
            results.append(r)
            label = path if len(path) <= 45 else '...' + path[-42:]
            print(
                f"{label:<46} {r['missing_level']*100:>12.2f}%  "
                f"{r['missing_counts']:.0f} / {r['total_counts']:.0f}"
            )
        except Exception as exc:
            print(f"  [ERROR] {path}: {exc}")

    # ── 汇总（多个文件时） ────────────────────────────────────────────────────
    if len(results) > 1:
        levels = [r['missing_level'] for r in results]
        print(f"\n{'='*70}")
        print(f"汇总（共 {len(results)} 个 sinogram）:")
        print(f"  平均 missing coincidence level : {np.mean(levels)*100:.2f}%")
        print(f"  最小                           : {np.min(levels)*100:.2f}%")
        print(f"  最大                           : {np.max(levels)*100:.2f}%")
        print(f"  标准差                         : {np.std(levels)*100:.4f}%")


if __name__ == '__main__':
    main()
