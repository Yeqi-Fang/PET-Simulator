#!/usr/bin/env python3
"""
listmode_to_incomplete.py

This script:
  1) Reads complete listmode data (.npz files)
  2) Filters out events involving detectors in specified angular ranges (creating "incomplete ring" data)
  3) Generates sinograms from the filtered data
  4) Creates comparison visualizations between complete and incomplete sinograms
  5) Saves both the incomplete listmode data and sinograms

Usage:
  python listmode_to_incomplete.py --input_dir /path/to/complete/listmode --output_dir /path/to/output --num_events 2000000000
"""

import os
import re
import glob
import numpy as np
import torch
import time
import argparse
import threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
from datetime import datetime

from pytomography.io.PET import gate

# 从generate_reconstruct.py导入可视化函数
try:
    from generate_reconstruct import visualize_sinogram
except ImportError:
    # 如果导入失败，则使用自定义版本
    def visualize_sinogram(sinogram, output_path, title="Sinogram Visualization", image_filename=None):
        """
        Create detailed visualizations for a single sinogram.
        
        Args:
            sinogram: The sinogram data (tensor or numpy array)
            output_path: Path to save the visualization
            title: Title for the visualization
            image_filename: Original image filename for reference
        """
        # 确保在使用非交互式后端
        matplotlib.use('Agg')
        
        try:
            # 确保sinogram是numpy数组
            if torch.is_tensor(sinogram):
                sinogram_np = sinogram.cpu().numpy()
            else:
                sinogram_np = sinogram
            
            sinogram_shape = sinogram_np.shape
            print(f"Sinogram shape: {sinogram_shape}")
            
            # 创建单切片视图 - 第一角度的前42个切片
            fig, ax = plt.subplots(figsize=(10, 6))
            slice_idx = min(42, sinogram_shape[2])
            im = ax.imshow(sinogram_np[0, :, :slice_idx], cmap='magma')
            
            # 设置标题和坐标轴
            if image_filename:
                ax.set_title(f"{title} ({image_filename})")
            else:
                ax.set_title(title)
            
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            print(f"  -> Saved sinogram visualization to {output_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create sinogram visualization: {e}")
            import traceback
            traceback.print_exc()

# 创建多切片正弦图可视化
def visualize_sinogram_multislice(sinogram, output_path, title="Sinogram Multislice", num_slices=6, image_filename=None):
    """
    Create a multi-slice view of a sinogram with slices evenly distributed through the volume.
    
    Args:
        sinogram: The sinogram data (tensor or numpy array)
        output_path: Path to save the visualization
        title: Title for the visualization
        num_slices: Number of slices to display
        image_filename: Original image filename for reference
    """
    # 确保使用非交互式后端
    matplotlib.use('Agg')
    
    try:
        # 确保sinogram是numpy数组
        if torch.is_tensor(sinogram):
            sinogram_np = sinogram.cpu().numpy()
        else:
            sinogram_np = sinogram
        
        # 获取形状信息
        sinogram_shape = sinogram_np.shape
        print(f"Generating multislice visualization for sinogram of shape {sinogram_shape}")
        
        # 计算要显示的切片
        depth = min(42, sinogram_shape[2])  # 使用前42个切片，如果更少则使用全部
        if num_slices > depth:
            num_slices = depth
            print(f"Adjusting number of slices to {num_slices} (maximum available)")
            
        # 选择均匀分布的切片索引
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
        
        # 创建绘图网格
        rows = (num_slices + 2) // 3  # 向上取整，确保足够的行
        cols = min(3, num_slices)  # 每行最多3个图
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])  # 确保axs是数组
        else:
            axs = axs.flatten()
        
        # 绘制每个切片
        for i, slice_idx in enumerate(slice_indices):
            if i < len(axs):
                im = axs[i].imshow(sinogram_np[:, :, slice_idx], cmap='magma', aspect='auto')
                axs[i].set_title(f'Ring Slice {slice_idx}')
                axs[i].set_xlabel('Radial Position')
                axs[i].set_ylabel('Angle')
        
        # 隐藏任何未使用的子图
        for i in range(num_slices, len(axs)):
            axs[i].set_visible(False)
        
        # 添加整体标题
        if image_filename:
            fig.suptitle(f"{title} ({image_filename})", fontsize=16)
        else:
            fig.suptitle(title, fontsize=16)
        
        # 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题留出空间
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"  -> Saved multislice sinogram visualization to {output_path}")
        
    except Exception as e:
        print(f"Warning: Failed to create multislice visualization: {e}")
        import traceback
        traceback.print_exc()

# PET scanner configuration (from main.py)
info = {
    'min_rsector_difference': np.float32(0.0),
    'crystal_length': np.float32(0.0),
    'radius': np.float32(253.71),
    'crystalTransNr': 13,
    'crystalTransSpacing': np.float32(4.01648),
    'crystalAxialNr': 7,
    'crystalAxialSpacing': np.float32(5.36556),
    'submoduleAxialNr': 1,
    'submoduleAxialSpacing': np.float32(0.0),
    'submoduleTransNr': 1,
    'submoduleTransSpacing': np.float32(0.0),
    'moduleTransNr': 1,
    'moduleTransSpacing': np.float32(0.0),
    'moduleAxialNr': 6,
    'moduleAxialSpacing': np.float32(37.55892),
    'rsectorTransNr': 28,
    'rsectorAxialNr': 1,
    'TOF': 0,
    'NrCrystalsPerRing': 364,  # 13 * 7 * 4
    'NrRings': 42,
    'firstCrystalAxis': 0
}

# Define missing angular sectors in degrees (start_deg, end_deg)
# Adjust these values to control how much of the ring is missing
MISSING_SECTORS = [(30, 90), (210, 270)]  # Example: 60-degree gaps at opposite sides
DEBUG = True

def build_missing_detector_ids(crystals_per_ring, num_rings, missing_sectors):
    """
    Build a set of detector IDs that lie in the specified missing angle ranges.

    ID is computed as: detector_id = ring_index * crystals_per_ring + crystal_index
    The angle for crystal_index is: (360 / crystals_per_ring) * crystal_index (degrees).

    Args:
        crystals_per_ring: Number of crystals in each ring
        num_rings: Number of axial rings
        missing_sectors: List of (start_angle, end_angle) tuples in degrees

    Returns:
        Set of detector IDs to be considered "missing"
    """
    missing_ids = set()
    if DEBUG:
        print("Building missing detector IDs...")
        print(f"  crystals_per_ring = {crystals_per_ring}, num_rings = {num_rings}")
        print(f"  missing_sectors = {missing_sectors}")

    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            angle_deg = (360.0 / crystals_per_ring) * crystal_idx
            # Check if angle_deg is in a missing sector
            is_missing = False
            for (deg_start, deg_end) in missing_sectors:
                if deg_start <= angle_deg <= deg_end:
                    is_missing = True
                    break

            if is_missing:
                det_id = ring_idx * crystals_per_ring + crystal_idx
                missing_ids.add(det_id)
                # Print some examples for debugging
                if DEBUG and len(missing_ids) < 10:
                    print(f"  Debug: ring={ring_idx}, crystal={crystal_idx}, "
                          f"angle={angle_deg:.2f} deg => det_id={det_id} (missing)")

    if DEBUG:
        print(f"Built missing detector set with {len(missing_ids)} detectors "
              f"({(len(missing_ids) / (crystals_per_ring * num_rings) * 100):.1f}% of total).")

    return missing_ids

def visualize_detector_coverage(crystals_per_ring, num_rings, missing_ids, missing_sectors, output_dir):
    """
    Create a visualization showing the missing detector coverage.
    
    Args:
        crystals_per_ring: Number of crystals in each ring
        num_rings: Number of axial rings
        missing_ids: Set of detector IDs considered missing
        missing_sectors: List of (start_angle, end_angle) tuples defining missing sectors
        output_dir: Directory to save the visualization
    """
    angles_active = []
    rings_active = []
    angles_missing = []
    rings_missing = []

    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            angle_deg = (360.0 / crystals_per_ring) * crystal_idx
            det_id = ring_idx * crystals_per_ring + crystal_idx
            if det_id in missing_ids:
                angles_missing.append(angle_deg)
                rings_missing.append(ring_idx)
            else:
                angles_active.append(angle_deg)
                rings_active.append(ring_idx)

    plt.figure(figsize=(10, 6))
    plt.scatter(angles_active, rings_active, s=2, c='blue', label='Active Detectors')
    plt.scatter(angles_missing, rings_missing, s=2, c='red', label='Missing Detectors')
    plt.title("Detector Coverage: Complete vs. Incomplete Ring")
    plt.xlabel("Azimuthal Angle (degrees)")
    plt.ylabel("Ring Index")
    
    # Highlight missing sectors
    for (deg_start, deg_end) in missing_sectors:
        plt.axvspan(deg_start, deg_end, color='red', alpha=0.1)
    
    plt.xlim(0, 360)
    plt.ylim(0, num_rings)
    plt.legend()
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "detector_coverage.png"), dpi=300)
    plt.close()
    print(f"Saved detector coverage visualization to {output_dir}/detector_coverage.png")

def filter_listmode_data(events, missing_ids):
    """
    Filter listmode data to remove events involving missing detectors.
    
    Args:
        events: Numpy array of detector ID pairs (can be structured array or 2D array)
        missing_ids: Set of detector IDs considered missing
    
    Returns:
        Filtered events array
    """
    # 检查数据格式并适应不同的格式
    if hasattr(events, 'dtype') and events.dtype.names is not None:
        # 处理结构化数组（如 .npz 文件中的数据）
        if 'det1_id' in events.dtype.names and 'det2_id' in events.dtype.names:
            # 直接访问具名字段
            det1_ids = events['det1_id']
            det2_ids = events['det2_id']
        else:
            # 尝试访问前两个字段
            field_names = events.dtype.names
            det1_ids = events[field_names[0]]
            det2_ids = events[field_names[1]]
    elif len(events.shape) >= 2 and events.shape[1] >= 2:
        # 处理标准二维数组
        det1_ids = events[:, 0]
        det2_ids = events[:, 1]
    else:
        raise ValueError(f"Unsupported event data format: shape={events.shape}, dtype={events.dtype}")
    
    # 构建布尔掩码，用于识别有效事件（两个探测器都不在missing_ids中）
    mask_missing_det1 = np.isin(det1_ids, list(missing_ids))
    mask_missing_det2 = np.isin(det2_ids, list(missing_ids))
    
    # 只保留两个探测器都不在缺失区域的事件
    valid_mask = ~(mask_missing_det1 | mask_missing_det2)
    
    # 根据原始数据类型返回适当格式的过滤数据
    if hasattr(events, 'dtype') and events.dtype.names is not None:
        # 返回结构化数组的过滤版本
        return events[valid_mask]
    elif len(events.shape) >= 2:
        # 返回二维数组的过滤版本
        return events[valid_mask]
    else:
        raise ValueError("Unable to filter events: unsupported data format")

def load_complete_sinogram(sinogram_dir, index, num_events):
    """
    Load the complete sinogram data from the given directory.
    
    Args:
        sinogram_dir: Base directory containing sinogram data
        index: Index number of the data
        num_events: Number of events
    
    Returns:
        Complete sinogram data if found, None otherwise
    """
    # 构造完整环正弦图文件路径
    complete_path = os.path.join(
        sinogram_dir, 
        f"reconstructed_index{index}_num{num_events}.npy"
    )
    
    # 尝试加载完整环正弦图
    if os.path.exists(complete_path):
        try:
            return np.load(complete_path)
        except Exception as e:
            print(f"Error loading complete sinogram from {complete_path}: {e}")
    
    return None

def process_and_compare_sinograms_background(complete_sinogram, incomplete_sinogram, output_dir, log_dir, image_index):
    """
    Process and compare complete and incomplete sinograms in a background thread.
    
    Args:
        complete_sinogram: Complete ring sinogram data (numpy array)
        incomplete_sinogram: Incomplete ring sinogram data (torch tensor or numpy array)
        output_dir: Output directory for saving results
        log_dir: Log directory for visualizations
        image_index: Index of the image being processed
    """
    def _process_and_compare():
        # 确保在线程中使用非交互式后端
        matplotlib.use('Agg')
        
        try:
            # 创建可视化目录
            vis_dir = os.path.join(log_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 确保incomplete_sinogram是numpy数组
            if torch.is_tensor(incomplete_sinogram):
                incomplete_sinogram_np = incomplete_sinogram.cpu().numpy()
            else:
                incomplete_sinogram_np = incomplete_sinogram
            
            # 获取形状信息
            incomplete_shape = incomplete_sinogram_np.shape
            complete_shape = complete_sinogram.shape
            
            print(f"Sinogram shapes - Complete: {complete_shape}, Incomplete: {incomplete_shape}")
            
            # 确保我们可以比较相同大小的数据
            if complete_shape != incomplete_shape:
                print(f"Warning: Sinogram shapes do not match. Resizing for visualization.")
                # 如果形状不同，可能需要进行裁剪或填充，这里简化处理
                min_shape = [min(s1, s2) for s1, s2 in zip(complete_shape, incomplete_shape)]
                incomplete_sinogram_np = incomplete_sinogram_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                complete_sinogram = complete_sinogram[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            # 计算差异图
            difference = complete_sinogram - incomplete_sinogram_np
            
            # 1. 单切片比较可视化
            fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
            
            # 选择第一个角度的第42个切片用于显示，或最大可用切片
            slice_idx = min(42, incomplete_shape[2]-1)
            
            # 完整环
            im1 = axs1[0].imshow(complete_sinogram[0, :, :slice_idx], cmap='magma')
            axs1[0].set_title(f'Complete Ring Sinogram (First {slice_idx} Slices)')
            axs1[0].axis('off')
            fig1.colorbar(im1, ax=axs1[0], fraction=0.046, pad=0.04)
            
            # 不完整环
            im2 = axs1[1].imshow(incomplete_sinogram_np[0, :, :slice_idx], cmap='magma')
            axs1[1].set_title(f'Incomplete Ring Sinogram (First {slice_idx} Slices)')
            axs1[1].axis('off')
            fig1.colorbar(im2, ax=axs1[1], fraction=0.046, pad=0.04)
            
            # 差异
            im3 = axs1[2].imshow(difference[0, :, :slice_idx], cmap='coolwarm')
            axs1[2].set_title(f'Difference (Complete - Incomplete)')
            axs1[2].axis('off')
            fig1.colorbar(im3, ax=axs1[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            fig1_filename = os.path.join(vis_dir, f"sinogram_comparison_index{image_index}.pdf")
            plt.savefig(fig1_filename, dpi=300)
            plt.close(fig1)
            print(f"  -> Saved sinogram comparison to {fig1_filename}")
            
            # 2. 多切片可视化对比
            fig2, axs2 = plt.subplots(2, 6, figsize=(20, 8))
            
            # 使用前42个切片，或者如果有更少的就用全部
            depth = min(42, incomplete_shape[2])
            slice_indices = np.linspace(0, depth-1, 6, dtype=int)
            
            # 完整环多切片
            for i, slice_idx in enumerate(slice_indices):
                axs2[0, i].imshow(complete_sinogram[:, :, slice_idx], cmap='magma', aspect='auto')
                axs2[0, i].set_title(f'Complete: Ring Slice {slice_idx}')
                if i == 0:
                    axs2[0, i].set_ylabel('Angle')
            
            # 不完整环多切片
            for i, slice_idx in enumerate(slice_indices):
                axs2[1, i].imshow(incomplete_sinogram_np[:, :, slice_idx], cmap='magma', aspect='auto')
                axs2[1, i].set_title(f'Incomplete: Ring Slice {slice_idx}')
                axs2[1, i].set_xlabel('Radial Position')
                if i == 0:
                    axs2[1, i].set_ylabel('Angle')
            
            plt.tight_layout()
            fig2_filename = os.path.join(vis_dir, f"sinogram_multislice_comparison_index{image_index}.pdf")
            plt.savefig(fig2_filename, dpi=300)
            plt.close(fig2)
            print(f"  -> Saved multi-slice comparison to {fig2_filename}")
            
            # 3. 详细的单切片对比
            fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
            
            # 选择切片索引
            middle_slice = min(20, depth-1)
            
            # 完整环详细切片
            complete_slice = complete_sinogram[:, :, middle_slice]
            im1 = axs3[0].imshow(complete_slice, cmap='magma', aspect='auto')
            axs3[0].set_title(f'Complete: Detailed Slice (Ring {middle_slice})')
            axs3[0].set_xlabel('Radial Position')
            axs3[0].set_ylabel('Angle')
            fig3.colorbar(im1, ax=axs3[0])
            
            # 不完整环详细切片
            incomplete_slice = incomplete_sinogram_np[:, :, middle_slice]
            im2 = axs3[1].imshow(incomplete_slice, cmap='magma', aspect='auto')
            axs3[1].set_title(f'Incomplete: Detailed Slice (Ring {middle_slice})')
            axs3[1].set_xlabel('Radial Position')
            axs3[1].set_ylabel('Angle')
            fig3.colorbar(im2, ax=axs3[1])
            
            # 差异详细切片
            diff_slice = complete_slice - incomplete_slice
            im3 = axs3[2].imshow(diff_slice, cmap='coolwarm', aspect='auto')
            axs3[2].set_title(f'Difference: Detailed Slice (Ring {middle_slice})')
            axs3[2].set_xlabel('Radial Position')
            axs3[2].set_ylabel('Angle')
            fig3.colorbar(im3, ax=axs3[2])
            
            plt.tight_layout()
            fig3_filename = os.path.join(vis_dir, f"sinogram_detailed_comparison_index{image_index}.pdf")
            plt.savefig(fig3_filename, dpi=300)
            plt.close(fig3)
            print(f"  -> Saved detailed slice comparison to {fig3_filename}")
            
            # 4. 生成一个缺失数据的热图，显示哪些区域缺失了数据
            try:
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                
                # 创建一个掩码，显示哪些区域缺失了数据
                missing_mask = np.abs(complete_sinogram.sum(axis=2) - incomplete_sinogram_np.sum(axis=2)) > 0.1
                
                # 将掩码转换为热图
                im4 = ax4.imshow(missing_mask, cmap='Reds', aspect='auto')
                ax4.set_title(f'Missing Data Regions (Red = Missing)')
                ax4.set_xlabel('Radial Position')
                ax4.set_ylabel('Angle')
                
                plt.tight_layout()
                fig4_filename = os.path.join(vis_dir, f"sinogram_missing_regions_index{image_index}.pdf")
                plt.savefig(fig4_filename, dpi=300)
                plt.close(fig4)
                print(f"  -> Saved missing regions visualization to {fig4_filename}")
            except Exception as e:
                print(f"Warning: Could not generate missing regions visualization: {e}")
            
        except Exception as e:
            print(f"Warning: Failed to generate sinogram comparison: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=_process_and_compare)
    thread.daemon = False  # 确保线程不会随主程序退出而终止
    thread.start()
    return thread

def process_listmode_file(input_file, output_dir, complete_sinogram_dir, log_dir, missing_ids, num_events, vis_level=2):
    """
    Process a single listmode file: filter it and generate a sinogram.
    
    Args:
        input_file: Path to the input listmode .npz file
        output_dir: Base output directory
        complete_sinogram_dir: Directory containing complete sinograms for comparison
        log_dir: Directory for visualizations and logs
        missing_ids: Set of detector IDs considered missing
        num_events: Event count for output path construction
        vis_level: Visualization detail level (0=minimal, 1=basic, 2=detailed)
    """
    # Extract index from filename using regex
    filename = os.path.basename(input_file)
    match = re.search(r'reconstructed_index(\d+)_num|listmode_data_minimal_(\d+)_', filename)
    if not match:
        print(f"Error: Could not parse index from filename: {filename}")
        return
    
    # 提取索引，考虑不同的文件命名模式
    index = match.group(1) if match.group(1) else match.group(2)
    print(f"\nProcessing listmode data for index {index}...")
    
    start_time = time.time()
    
    # Load listmode data
    try:
        data = np.load(input_file)
        
        # 根据数据类型提取事件
        if isinstance(data, np.ndarray):
            # 如果直接加载为数组
            events_data = data
        else:
            # 如果是.npz文件
            if 'listmode' in data:
                events_data = data['listmode']
            else:
                # 尝试获取第一个数组
                try:
                    events_data = next(iter(data.values()))
                except:
                    # 直接打印数据内容以便调试
                    print(f"Data keys: {list(data.keys())}")
                    raise ValueError(f"Cannot extract event data from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return
    
    print(f"Loaded events from {input_file}, format: shape={events_data.shape}, dtype={events_data.dtype}")
    
    # 检查事件数量
    if hasattr(events_data, 'shape'):
        if len(events_data.shape) == 1 and hasattr(events_data, 'dtype') and events_data.dtype.names is not None:
            # 结构化数组
            num_loaded_events = len(events_data)
        else:
            # 常规数组
            num_loaded_events = events_data.shape[0]
    else:
        num_loaded_events = "unknown"
    
    print(f"Loaded {num_loaded_events} events from {input_file}")
    
    # Filter events to create incomplete ring data
    try:
        filtered_events = filter_listmode_data(events_data, missing_ids)
        
        # 获取过滤后的事件数量
        if hasattr(filtered_events, 'shape'):
            if len(filtered_events.shape) == 1 and hasattr(filtered_events, 'dtype') and filtered_events.dtype.names is not None:
                num_filtered_events = len(filtered_events)
            else:
                num_filtered_events = filtered_events.shape[0]
        else:
            num_filtered_events = "unknown"
            
        print(f"Filtered to {num_filtered_events} events ({float(num_filtered_events)/float(num_loaded_events)*100:.1f}% of original)")
    except Exception as e:
        print(f"Error filtering events: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if isinstance(num_filtered_events, (int, float)) and num_filtered_events == 0:
        print("Warning: All events were filtered out. Check missing sectors configuration.")
        return
    
    # Set up output directories
    incomplete_lm_dir = os.path.join(output_dir, 'listmode_incomplete')
    incomplete_sinogram_dir = os.path.join(output_dir, 'sinogram_incomplete')
    vis_dir = os.path.join(log_dir, "visualizations")
    
    os.makedirs(incomplete_lm_dir, exist_ok=True)
    os.makedirs(incomplete_sinogram_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save filtered listmode data in background
    out_lm_path = os.path.join(incomplete_lm_dir, f"incomplete_index{index}_num{num_events}.npz")
    save_thread = threading.Thread(
        target=lambda: np.savez_compressed(out_lm_path, listmode=filtered_events)
    )
    save_thread.daemon = False
    save_thread.start()
    
    # Convert filtered events to detector IDs for sinogram generation
    # 确保数据是适当的形式传递给gate.listmode_to_sinogram
    try:
        # 为listmode_to_sinogram转换数据格式
        if hasattr(filtered_events, 'dtype') and filtered_events.dtype.names is not None:
            # 从结构化数组创建适当的torch张量
            if 'det1_id' in filtered_events.dtype.names and 'det2_id' in filtered_events.dtype.names:
                detector_ids = torch.tensor(
                    np.column_stack((filtered_events['det1_id'], filtered_events['det2_id'])),
                    dtype=torch.int32
                )
            else:
                # 使用前两个字段
                field_names = filtered_events.dtype.names
                detector_ids = torch.tensor(
                    np.column_stack((filtered_events[field_names[0]], filtered_events[field_names[1]])),
                    dtype=torch.int32
                )
        else:
            # 假设已经是二维数组
            detector_ids = torch.from_numpy(filtered_events[:, :2]).to(torch.int32)
            
        # 生成正弦图
        print("Generating sinogram from incomplete data...")
        incomplete_sinogram = gate.listmode_to_sinogram(detector_ids, info)
        
        # 保存不完整正弦图
        out_sinogram_path = os.path.join(incomplete_sinogram_dir, f"incomplete_index{index}_num{num_events}")
        np.save(out_sinogram_path, incomplete_sinogram.numpy().astype(np.float32))
        print(f"Saved incomplete sinogram to {out_sinogram_path}")
        
        # 创建不完整正弦图可视化
        if vis_level >= 1:
            # 1. 标准单视图可视化 (类似于generate_reconstruct.py中的版本)
            standard_vis_path = os.path.join(vis_dir, f"sinogram_{index}.pdf")
            visualize_sinogram(
                sinogram=incomplete_sinogram,
                output_path=standard_vis_path,
                title="Incomplete Ring Sinogram",
                image_filename=f"3d_image_{index}.npy"
            )
            
            # 2. 多切片可视化
            multislice_vis_path = os.path.join(vis_dir, f"sinogram_multislice_3d_image_{index}.npy.pdf")
            visualize_sinogram_multislice(
                sinogram=incomplete_sinogram,
                output_path=multislice_vis_path,
                title="Incomplete Ring Sinogram Multislice",
                num_slices=6,
                image_filename=f"3d_image_{index}.npy"
            )
        
        # 尝试加载对应的完整环正弦图进行比较
        complete_sinogram = load_complete_sinogram(complete_sinogram_dir, index, num_events)
        
        # 如果找到完整环正弦图，创建可视化对比
        if complete_sinogram is not None and vis_level >= 2:
            print("Found matching complete sinogram, generating comparison...")
            comparison_thread = process_and_compare_sinograms_background(
                complete_sinogram=complete_sinogram,
                incomplete_sinogram=incomplete_sinogram,
                output_dir=output_dir,
                log_dir=log_dir,
                image_index=index
            )
        
    except Exception as e:
        print(f"Error generating sinogram: {e}")
        import traceback
        traceback.print_exc()
    
    # 等待后台保存线程完成
    save_thread.join()
    print(f"Completed processing index {index} in {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Convert complete listmode data to incomplete ring data')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing complete listmode .npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--sinogram_dir', type=str, 
                        help='Directory containing complete sinograms for comparison')
    parser.add_argument('--num_events', type=int, required=True,
                        help='Number of events (used for path construction)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate detector coverage visualization')
    parser.add_argument('--vis_level', type=int, default=2, choices=[0, 1, 2],
                        help='Visualization detail level (0=minimal, 1=basic, 2=detailed)')
    parser.add_argument('--missing_start1', type=float, default=30,
                        help='Start angle of first missing sector (degrees)')
    parser.add_argument('--missing_end1', type=float, default=90,
                        help='End angle of first missing sector (degrees)')
    parser.add_argument('--missing_start2', type=float, default=210,
                        help='Start angle of second missing sector (degrees)')
    parser.add_argument('--missing_end2', type=float, default=270,
                        help='End angle of second missing sector (degrees)')
    parser.add_argument('--missing_start3', type=float, default=210,
                        help='Start angle of second missing sector (degrees)')
    parser.add_argument('--missing_end3', type=float, default=270,
                        help='End angle of second missing sector (degrees)')    
    parser.add_argument('--missing_start4', type=float, default=210,
                        help='Start angle of second missing sector (degrees)')
    parser.add_argument('--missing_end4', type=float, default=270,
                        help='End angle of second missing sector (degrees)')   
    
    args = parser.parse_args()
    
    # 根据命令行参数设置缺失扇区
    missing_sectors = [
        (args.missing_start1, args.missing_end1),
        (args.missing_start2, args.missing_end2),
        (args.missing_start3, args.missing_end3),
        (args.missing_start4, args.missing_end4)
    ]
    
    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(args.output_dir, 'log_incomplete', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # If sinogram_dir not provided, use default location
    sinogram_dir = args.sinogram_dir
    if not sinogram_dir:
        # 尝试构造默认位置：与输入目录平级的sinogram目录
        parent_dir = os.path.dirname(args.input_dir)
        sinogram_dir = os.path.join(parent_dir, 'sinogram')
        if not os.path.exists(sinogram_dir):
            print(f"Complete sinogram directory not found at {sinogram_dir}")
            print("Visualizations will not include comparisons")
            sinogram_dir = None
    
    # Build missing detector IDs set
    missing_ids = build_missing_detector_ids(
        crystals_per_ring=info['NrCrystalsPerRing'],
        num_rings=info['NrRings'],
        missing_sectors=missing_sectors
    )
    
    # Generate detector coverage visualization
    if args.visualize:
        visualize_detector_coverage(
            crystals_per_ring=info['NrCrystalsPerRing'],
            num_rings=info['NrRings'],
            missing_ids=missing_ids,
            missing_sectors=missing_sectors,
            output_dir=log_dir
        )
    
    # Find all listmode files
    if os.path.isdir(args.input_dir):
        # Look for .npz files in the directory
        listmode_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    else:
        # Treat input_dir as a single file
        listmode_files = [args.input_dir] if os.path.exists(args.input_dir) else []
    
    print(f"Found {len(listmode_files)} listmode files to process")
    
    if not listmode_files:
        print("No input files found. Check the --input_dir path.")
        return
    
    # 跟踪所有后台线程
    all_threads = []
    
    # Process each file
    for i, lm_file in enumerate(listmode_files):
        print(f"\n[{i+1}/{len(listmode_files)}] Processing {lm_file}")
        
        # 处理单个文件
        process_listmode_file(
            input_file=lm_file,
            output_dir=args.output_dir,
            complete_sinogram_dir=sinogram_dir,
            log_dir=log_dir,
            missing_ids=missing_ids,
            num_events=args.num_events,
            vis_level=args.vis_level
        )
    
    # 等待所有后台线程完成
    for thread in all_threads:
        thread.join()
    
    print("\nAll files processed. Incomplete ring data generation complete.")

if __name__ == "__main__":
    main()