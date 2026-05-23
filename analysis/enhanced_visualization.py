#!/usr/bin/env python3
"""
enhanced_visualization.py

用于创建高质量的正弦图可视化工具，支持多角度、多切片和对比分析。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import os

def visualize_sinogram_multislice(sinogram, output_path=None, title="Sinogram Visualization", 
                                  num_slices=4, cmap='magma', figsize=(12, 8)):
    """
    创建多切片正弦图可视化，展示多个轴向切片
    
    Args:
        sinogram: 3D正弦图数据，形状为(views, bins, slices)
        output_path: 保存图像的路径，如果为None则显示而不保存
        title: 图像标题
        num_slices: 要显示的切片数量
        cmap: 颜色映射
        figsize: 图像尺寸
    """
    # 确保sinogram是numpy数组
    if not isinstance(sinogram, np.ndarray):
        sinogram = np.array(sinogram)
    
    # 计算切片间隔
    total_slices = sinogram.shape[2]
    if total_slices < num_slices:
        num_slices = total_slices
    
    # 选择均匀分布的切片
    slice_indices = np.linspace(0, total_slices-1, num_slices, dtype=int)
    
    # 创建图形
    fig, axes = plt.subplots(1, num_slices, figsize=figsize)
    if num_slices == 1:
        axes = [axes]
    
    # 计算全局数据范围以保持一致的颜色映射
    vmin = np.min(sinogram)
    vmax = np.max(sinogram)
    
    # 对每个切片进行可视化
    for i, slice_idx in enumerate(slice_indices):
        im = axes[i].imshow(sinogram[:, :, slice_idx], cmap=cmap, aspect='auto', 
                           vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].set_xlabel('Bin')
        if i == 0:
            axes[i].set_ylabel('View')
    
    # 添加颜色条
    fig.colorbar(im, ax=axes, shrink=0.8, label='Counts')
    
    # 设置整体标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_sinogram_multi_perspective(sinogram, output_path=None, title="Sinogram Perspectives", 
                                        cmap='magma', figsize=(15, 10)):
    """
    创建多角度正弦图可视化，展示不同视角的投影数据
    
    Args:
        sinogram: 3D正弦图数据，形状为(views, bins, slices)
        output_path: 保存图像的路径，如果为None则显示而不保存
        title: 图像标题
        cmap: 颜色映射
        figsize: 图像尺寸
    """
    # 确保sinogram是numpy数组
    if not isinstance(sinogram, np.ndarray):
        sinogram = np.array(sinogram)
    
    # 获取维度
    views, bins, slices = sinogram.shape
    
    # 创建图形布局
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])
    
    # 创建中间切片的视图-bin图
    ax1 = fig.add_subplot(gs[0, 0])
    mid_slice = slices // 2
    im1 = ax1.imshow(sinogram[:, :, mid_slice], cmap=cmap, aspect='auto')
    ax1.set_title(f'View-Bin (Slice {mid_slice})')
    ax1.set_xlabel('Bin')
    ax1.set_ylabel('View')
    
    # 创建中间bin的视图-切片图
    ax2 = fig.add_subplot(gs[0, 1])
    mid_bin = bins // 2
    im2 = ax2.imshow(sinogram[:, mid_bin, :], cmap=cmap, aspect='auto')
    ax2.set_title(f'View-Slice (Bin {mid_bin})')
    ax2.set_xlabel('Slice')
    ax2.set_ylabel('View')
    
    # 创建中间视图的bin-切片图
    ax3 = fig.add_subplot(gs[1, 0])
    mid_view = views // 2
    im3 = ax3.imshow(sinogram[mid_view, :, :], cmap=cmap, aspect='auto')
    ax3.set_title(f'Bin-Slice (View {mid_view})')
    ax3.set_xlabel('Slice')
    ax3.set_ylabel('Bin')
    
    # 创建3D图
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    X, Y = np.meshgrid(range(bins), range(views))
    Z = sinogram[:, :, mid_slice]
    
    # 归一化数据以获得更好的可视化效果
    norm = colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    
    # 创建3D表面图
    surf = ax4.plot_surface(X, Y, Z, cmap=cmap, norm=norm, 
                           linewidth=0, antialiased=True, alpha=0.8)
    ax4.set_title(f'3D Surface (Slice {mid_slice})')
    ax4.set_xlabel('Bin')
    ax4.set_ylabel('View')
    ax4.set_zlabel('Counts')
    
    # 添加颜色条
    cbar_ax = fig.add_subplot(gs[:, 2])
    fig.colorbar(im1, cax=cbar_ax, label='Counts')
    
    # 设置整体标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_sinograms(complete_sinogram, incomplete_sinogram, output_path=None, 
                     title="Complete vs Incomplete Sinogram", num_slices=3, 
                     cmap='magma', figsize=(15, 12)):
    """
    创建完整环与不完整环正弦图的对比可视化
    
    Args:
        complete_sinogram: 完整环的3D正弦图数据
        incomplete_sinogram: 不完整环的3D正弦图数据
        output_path: 保存图像的路径，如果为None则显示而不保存
        title: 图像标题
        num_slices: 要显示的切片数量
        cmap: 颜色映射
        figsize: 图像尺寸
    """
    # 确保正弦图是numpy数组
    if not isinstance(complete_sinogram, np.ndarray):
        complete_sinogram = np.array(complete_sinogram)
    if not isinstance(incomplete_sinogram, np.ndarray):
        incomplete_sinogram = np.array(incomplete_sinogram)
    
    # 计算差异
    difference = complete_sinogram - incomplete_sinogram
    
    # 计算相对差异百分比（避免除以零）
    epsilon = 1e-10  # 小常数以避免除以零
    relative_diff = np.zeros_like(difference)
    mask = (complete_sinogram > epsilon)
    relative_diff[mask] = (difference[mask] / (complete_sinogram[mask] + epsilon)) * 100
    
    # 计算切片间隔
    total_slices = complete_sinogram.shape[2]
    if total_slices < num_slices:
        num_slices = total_slices
    
    # 选择均匀分布的切片
    slice_indices = np.linspace(0, total_slices-1, num_slices, dtype=int)
    
    # 创建图形布局
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, num_slices+1, figure=fig, width_ratios=[1]*num_slices + [0.05])
    
    # 计算每个数据集的全局范围
    vmin_complete = np.min(complete_sinogram)
    vmax_complete = np.max(complete_sinogram)
    
    vmin_incomplete = np.min(incomplete_sinogram)
    vmax_incomplete = np.max(incomplete_sinogram)
    
    # 使用统一的颜色范围
    vmin = min(vmin_complete, vmin_incomplete)
    vmax = max(vmax_complete, vmax_incomplete)
    
    # 差异图的颜色范围
    vmin_diff = np.min(difference)
    vmax_diff = np.max(difference)
    
    # 相对差异的颜色范围（对称化）
    abs_max_rel = max(abs(np.min(relative_diff)), abs(np.max(relative_diff)))
    vmin_rel = -abs_max_rel
    vmax_rel = abs_max_rel
    
    # 对每个切片进行可视化
    images = []
    for row, (data, title_prefix, v_range) in enumerate([
        (complete_sinogram, "Complete", (vmin, vmax)),
        (incomplete_sinogram, "Incomplete", (vmin, vmax)),
        (difference, "Difference", (vmin_diff, vmax_diff))
    ]):
        for i, slice_idx in enumerate(slice_indices):
            ax = fig.add_subplot(gs[row, i])
            im = ax.imshow(data[:, :, slice_idx], cmap=cmap, aspect='auto', 
                          vmin=v_range[0], vmax=v_range[1])
            ax.set_title(f'{title_prefix} (Slice {slice_idx})')
            if i == 0:
                ax.set_ylabel('View')
            if row == 2:
                ax.set_xlabel('Bin')
            images.append(im)
    
    # 添加颜色条
    cbar_axes = [fig.add_subplot(gs[i, -1]) for i in range(3)]
    for i, (im, label) in enumerate(zip(
        [images[0], images[num_slices], images[2*num_slices]], 
        ['Counts (Complete)', 'Counts (Incomplete)', 'Difference']
    )):
        fig.colorbar(im, cax=cbar_axes[i], label=label)
    
    # 设置整体标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_detector_coverage_3d(crystals_per_ring, num_rings, missing_ids, output_path=None,
                                  title="3D Detector Coverage", figsize=(12, 10)):
    """
    创建3D探测器覆盖可视化，显示缺失的探测器元素
    
    Args:
        crystals_per_ring: 每环的晶体数量
        num_rings: 环数
        missing_ids: 缺失探测器ID集合
        output_path: 保存图像的路径，如果为None则显示而不保存
        title: 图像标题
        figsize: 图像尺寸
    """
    # 创建3D图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算所有探测器的位置
    all_rings = []
    all_angles = []
    all_ids = []
    missing_rings = []
    missing_angles = []
    
    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            det_id = ring_idx * crystals_per_ring + crystal_idx
            angle_rad = (2 * np.pi / crystals_per_ring) * crystal_idx
            
            # 将ID添加到适当的列表
            if det_id in missing_ids:
                missing_rings.append(ring_idx)
                missing_angles.append(angle_rad)
            else:
                all_rings.append(ring_idx)
                all_angles.append(angle_rad)
                all_ids.append(det_id)
    
    # 转换为笛卡尔坐标
    radius = 1.0  # 单位半径
    
    # 活跃探测器
    x_active = radius * np.cos(all_angles)
    y_active = radius * np.sin(all_angles)
    z_active = np.array(all_rings)
    
    # 缺失探测器
    x_missing = radius * np.cos(missing_angles)
    y_missing = radius * np.sin(missing_angles)
    z_missing = np.array(missing_rings)
    
    # 绘制活跃探测器（蓝色）
    ax.scatter(x_active, y_active, z_active, c='blue', marker='o', alpha=0.6, 
              label='Active Detectors')
    
    # 绘制缺失探测器（红色）
    ax.scatter(x_missing, y_missing, z_missing, c='red', marker='x', alpha=0.8, 
              label='Missing Detectors')
    
    # 添加一个参考圆柱体
    theta = np.linspace(0, 2*np.pi, 100)
    for z in np.linspace(0, num_rings-1, 5):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z_circle = np.ones_like(theta) * z
        ax.plot(x, y, z_circle, color='gray', alpha=0.3)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Ring')
    ax.set_title(title)
    
    # 添加图例
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # 测试代码
    # 创建模拟的正弦图数据
    views, bins, slices = 180, 200, 64
    test_sinogram = np.random.random((views, bins, slices)) * 100
    
    # 测试多切片可视化
    visualize_sinogram_multislice(test_sinogram, "test_multislice.png", 
                                 title="Test Multislice Visualization")
    
    # 测试多角度可视化
    visualize_sinogram_multi_perspective(test_sinogram, "test_multi_perspective.png")
    
    # 测试对比可视化
    incomplete_sinogram = test_sinogram * np.random.random((views, bins, slices)) * 0.8
    compare_sinograms(test_sinogram, incomplete_sinogram, "test_comparison.png")
    
    # 测试3D探测器覆盖可视化
    missing_ids = set(range(0, 1000, 3))  # 模拟缺失探测器
    visualize_detector_coverage_3d(180, 32, missing_ids, "test_detector_3d.png")