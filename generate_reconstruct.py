#!/usr/bin/env python3
# main.py - Optimized PET simulation and reconstruction pipeline

import os
import re
import glob
import numpy as np
import torch
import time
import threading
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 重要：在导入matplotlib之前设置后端为Agg（非交互式，线程安全）
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt

from pytomography.io.PET import gate, shared

from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator
from pet_simulator.utils import save_events, save_events_binary
from reconstruction_all import reconstruct_volume_for_lmf
from outlier_detection import (
    global_outlier_detection,
    local_outlier_detection,
    edge_outlier_detection,
    combined_outlier_detection,
    analyze_outlier_masks,
    remove_outliers_iteratively
)

# PET scanner configuration
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

# Reconstruction parameters
voxel_size = 2.78
volume_size = 128
extended_size = 128
n_iters = 1
n_subsets = 34
psf_fwhm_mm = 4.5
outlier = False
num_events = int(8e9)
save_events_pos = False

def save_events_background(file_path, events, save_full_data=False):
    """Save events in a background thread to avoid blocking the main process."""
    thread = threading.Thread(
        target=save_events_binary,
        args=(file_path, events, save_full_data)
    )
    thread.daemon = False  # Ensure thread doesn't exit when main program exits
    thread.start()
    return thread

def process_and_save_sinogram_background(events, info, output_path, log_dir=None, image_filename=None):
    """
    Generate sinogram from events and save it in a background thread.
    
    Args:
        events: The detector events data
        info: PET scanner info
        output_path: Path to save the sinogram
        log_dir: Optional directory to save visualization
        image_filename: Original image filename for visualization
    """
    def _process_and_save():
        # 确保在线程中使用非交互式后端
        matplotlib.use('Agg')
        
        # Convert events to detector IDs
        detector_ids = torch.from_numpy(events).to(torch.int32)
        
        # Generate sinogram
        sinogram = gate.listmode_to_sinogram(detector_ids, info)
        
        # Cleanup detector_ids to save memory
        del detector_ids
        
        # Optional visualization
        if log_dir is not None and image_filename is not None:
            try:
                # 获取正弦图形状和数据
                sinogram_np = sinogram.numpy()
                sinogram_shape = sinogram_np.shape
                print(f"Sinogram shape: {sinogram_shape}")
                
                # 1. 创建原始的单切片正弦图可视化（保持原有功能）
                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
                im0 = ax1.imshow(sinogram_np[0, :, :42], cmap='magma')
                ax1.set_title(f'Original Sinogram (First 42 Slices)')
                ax1.axis('off')
                fig1.colorbar(im0, ax=ax1, fraction=0.046, pad=0.04)
                plt.tight_layout()
                
                # 保存原始的单切片可视化
                fig1_filename = os.path.join(log_dir, f"sinogram_{image_filename}.pdf")
                plt.savefig(fig1_filename, dpi=300)
                plt.close(fig1)
                print(f"  -> Saved original sinogram figure to {fig1_filename}")
                
                # 2. 创建多切片正弦图视图 (固定第一维，显示不同的第三维切片)
                fig2, axs = plt.subplots(2, 3, figsize=(15, 8))
                axs = axs.flatten()
                
                # 选择6个均匀分布的切片索引
                depth = min(42, sinogram_np.shape[2])  # 使用前42个切片，如果有更少就用全部
                slice_indices = np.linspace(0, depth-1, 6, dtype=int)
                
                # 固定第一维为0，遍历第三维的不同切片
                for i, slice_idx in enumerate(slice_indices):
                    # 使用[:, :, slice_idx]索引得到2D切片
                    slice_data = sinogram_np[:, :, slice_idx]
                    axs[i].imshow(slice_data, cmap='magma', aspect='auto')
                    axs[i].set_title(f'Ring Slice {slice_idx}')
                    axs[i].set_xlabel('Radial Position')
                    axs[i].set_ylabel('Angle')
                
                plt.tight_layout()
                fig2_filename = os.path.join(log_dir, f"sinogram_multislice_{image_filename}.pdf")
                plt.savefig(fig2_filename, dpi=300)
                plt.close(fig2)
                print(f"  -> Saved multi-slice sinogram to {fig2_filename}")
                
                # # 3. 创建不同视角的切片可视化
                # fig3, axs = plt.subplots(1, 2, figsize=(16, 6))
                
                # # 取第一个角度的切片(固定第一维为0)，显示径向-环差视图
                # middle_radial = sinogram_np.shape[1] // 2
                # radial_slice = sinogram_np[0, middle_radial, :depth]  # 取中间径向位置
                
                # # 创建2D网格以显示1D数据
                # x = np.arange(len(radial_slice))
                # X, Y = np.meshgrid(x, np.array([0]))
                
                # # 以伪彩色方式显示1D数据
                # im1 = axs[0].scatter(X, Y, c=radial_slice, cmap='magma', s=50)
                # axs[0].set_title(f'Radial-Ring Slice (Middle Radial Position)')
                # axs[0].set_xlabel('Ring Difference')
                # axs[0].set_yticks([])  # 隐藏Y轴刻度
                # axs[0].set_ylim(-0.5, 0.5)  # 固定Y轴范围
                # fig3.colorbar(im1, ax=axs[0])
                
                # # 显示角度-环差视图
                # middle_angle = sinogram_np.shape[0] // 2
                # angle_slice = sinogram_np[:, :, 0].T  # 转置使角度在水平轴上
                # im2 = axs[1].imshow(angle_slice, cmap='magma', aspect='auto')
                # axs[1].set_title(f'Angle-Radial View (First Ring)')
                # axs[1].set_xlabel('Angle')
                # axs[1].set_ylabel('Radial Position')
                # fig3.colorbar(im2, ax=axs[1])
                
                # plt.tight_layout()
                # fig3_filename = os.path.join(log_dir, f"sinogram_side_views_{image_filename}.pdf")
                # plt.savefig(fig3_filename, dpi=300)
                # plt.close(fig3)
                # print(f"  -> Saved side-view sinogram to {fig3_filename}")
                
                # 4. 创建单个第三维切片的详细视图
                fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
                middle_slice = min(20, depth-1)  # 选择第20个切片或最大可用切片
                slice_data = sinogram_np[:, :, middle_slice]
                im4 = ax4.imshow(slice_data, cmap='magma', aspect='auto')
                ax4.set_title(f'Detailed Sinogram Slice (Ring Difference = {middle_slice})')
                ax4.set_xlabel('Radial Position')
                ax4.set_ylabel('Angle')
                fig4.colorbar(im4)
                plt.tight_layout()
                
                fig4_filename = os.path.join(log_dir, f"sinogram_detailed_slice_{image_filename}.pdf")
                plt.savefig(fig4_filename, dpi=300)
                plt.close(fig4)
                print(f"  -> Saved detailed slice visualization to {fig4_filename}")
                
            except Exception as e:
                print(f"Warning: Failed to save sinogram visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Save sinogram
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, sinogram_np.astype(np.float32))
        
        # Cleanup to save memory
        del sinogram
        print(f"  -> Saved sinogram to {output_path}")
    
    thread = threading.Thread(target=_process_and_save)
    thread.daemon = False
    thread.start()
    return thread

def create_comparison_visualizations(image, result_3d, log_dir, image_filename):
    """Create and save comparison visualizations between original and reconstructed images."""
    # Create visualizations in a background thread
    def _create_visualizations():
        # 确保在线程中使用非交互式后端
        matplotlib.use('Agg')
        
        try:
            # Axial slice (z-axis)
            slice_index = result_3d.shape[2] // 2
            fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
            im0 = axs[0].imshow(image[:, :, slice_index], cmap='magma', interpolation='nearest')
            axs[0].set_title(f'Original Image Slice (z = {slice_index})')
            axs[0].axis('off')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            
            im1 = axs[1].imshow(result_3d[:, :, slice_index], cmap='magma', interpolation='nearest')
            axs[1].set_title(f'Reconstructed Slice (z = {slice_index})')
            axs[1].axis('off')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            fig_filename = os.path.join(log_dir, f"comparison_{image_filename}.pdf")
            plt.savefig(fig_filename, dpi=300)
            plt.close(fig)
            
            # Coronal slice (x-axis)
            slice_index_x = result_3d.shape[0] // 2
            fig2, axs2 = plt.subplots(1, 2, figsize=(15, 6))
            im0 = axs2[0].imshow(image[slice_index_x, :, :], cmap='magma', interpolation='nearest')
            axs2[0].set_title(f'Original Image Slice (x = {slice_index_x})')
            axs2[0].axis('off')
            fig2.colorbar(im0, ax=axs2[0], fraction=0.046, pad=0.04)
            
            im1 = axs2[1].imshow(result_3d[slice_index_x, :, :], cmap='magma', interpolation='nearest')
            axs2[1].set_title(f'Reconstructed Slice (x = {slice_index_x})')
            axs2[1].axis('off')
            fig2.colorbar(im1, ax=axs2[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            fig2_filename = os.path.join(log_dir, f"comparison_x_{image_filename}.pdf")
            plt.savefig(fig2_filename, dpi=300)
            plt.close(fig2)
            
            # Sagittal slice (y-axis)
            slice_index_y = result_3d.shape[1] // 2
            fig3, axs3 = plt.subplots(1, 2, figsize=(15, 4.5))
            im0 = axs3[0].imshow(image[:, slice_index_y, :], cmap='magma', interpolation='nearest')
            axs3[0].set_title(f'Original Image Slice (y = {slice_index_y})')
            axs3[0].axis('off')
            fig3.colorbar(im0, ax=axs3[0], fraction=0.046, pad=0.04)
            
            im1 = axs3[1].imshow(result_3d[:, slice_index_y, :], cmap='magma', interpolation='nearest')
            axs3[1].set_title(f'Reconstructed Slice (y = {slice_index_y})')
            axs3[1].axis('off')
            fig3.colorbar(im1, ax=axs3[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            fig3_filename = os.path.join(log_dir, f"comparison_y_{image_filename}.pdf")
            plt.savefig(fig3_filename, dpi=300)
            plt.close(fig3)
            
            print(f"  -> Saved comparison visualizations to {log_dir}")
        except Exception as e:
            print(f"Warning: Failed to create comparison visualizations: {e}")
    
    thread = threading.Thread(target=_create_visualizations)
    thread.daemon = False
    thread.start()
    return thread

def main():
    """Main function to simulate PET events, reconstruct volumes, and save results."""
    # 在主函数开始时设置全局matplotlib后端为Agg
    matplotlib.use('Agg')
    
    # Create PET scanner geometry from info
    geometry = create_pet_geometry(info)

    # Save the detector lookup table once
    dummy_image = np.ones((1, 1, 1), dtype=np.float32)
    simulator_for_lut = PETSimulator(geometry, dummy_image, voxel_size=2.78)
    simulator_for_lut.save_detector_positions("detector_lut.txt")
    lut_file = 'detector_lut.txt'

    # Define directories
    # base_dir = "data/dataset/train_npy_crop"
    # lmf_output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/listmode'
    # output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}'
    # output_dir_sinogram = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/sinogram'

    base_dir = "../gratuate-thesis/data/dataset/train_npy_crop"
    lmf_output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/listmode'
    output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}'
    output_dir_sinogram = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/sinogram'
    
    # Create directories
    os.makedirs(lmf_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_sinogram, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # Create and cache the scanner LUT for reuse
    lut_data = np.loadtxt(lut_file, skiprows=1)
    scanner_lut_np = lut_data[:, 1:4]
    scanner_lut = torch.from_numpy(scanner_lut_np).float()
    
    # Set up thread pool for managing parallel operations
    max_parallel_processes = min(16, os.cpu_count())
    print(f"Using up to {max_parallel_processes} parallel threads for I/O operations")
    
    # Keep track of all background threads
    all_threads = []
    
    # Process each image file
    for i in range(0, 170):

        # start time 
        t_start_total = time.time()

        image_filename = f"3d_image_{i}.npy"
        image_path = os.path.join(base_dir, image_filename)
        print(f"\nProcessing {image_filename} ...")
        
        # Load the 3D image
        image = np.load(image_path)
        print(f"Image shape: {image.shape}")
        
        # Create simulator and generate events
        simulator = PETSimulator(geometry, image, voxel_size=2.78, save_events_pos=save_events_pos)
        
        print("Starting simulation...")
        start_time = time.time()
        events = simulator.simulate_events(num_events=num_events, use_multiprocessing=True)
        simulation_time = time.time() - start_time
        print(f"Generated {len(events)} valid events in {simulation_time:.2f} seconds.")
        
        # Generate output filenames
        minimal_file = os.path.join(lmf_output_dir, f"listmode_data_minimal_{i}_{num_events}")
        out_name = f"reconstructed_index{i}_num{num_events}"
        out_path = os.path.join(output_dir, out_name)
        out_path_sinogram = os.path.join(output_dir_sinogram, out_name)
        
        # Start background processes for saving events and generating sinogram
        events_thread = save_events_background(minimal_file, events, save_full_data=False)
        all_threads.append(events_thread)
        
        sinogram_thread = process_and_save_sinogram_background(
            events=events,
            info=info,
            output_path=out_path_sinogram,
            log_dir=log_dir,
            image_filename=image_filename
        )
        all_threads.append(sinogram_thread)
        
        # Reconstruct volume directly using events
        # print("Starting reconstruction...")
        # start_time = time.time()
        # result_3d = reconstruct_volume_for_lmf(
        #     detector_ids_np=events,
        #     scanner_lut=scanner_lut,
        #     voxel_size=voxel_size,
        #     volume_size=volume_size,
        #     extended_size=extended_size,
        #     n_iters=n_iters,
        #     n_subsets=n_subsets,
        #     psf_fwhm_mm=psf_fwhm_mm,
        #     detector_outlier=outlier
        # )
        # reconstruction_time = time.time() - start_time
        # print(f"Reconstruction completed in {reconstruction_time:.2f} seconds.")
        
        # Save reconstructed volume
        # np.save(out_path, result_3d)
        # print(f"  -> Saved reconstructed volume to {out_path}")
        
        # Create visualizations in background
        # vis_thread = create_comparison_visualizations(
        #     image=image,
        #     result_3d=result_3d,
        #     log_dir=log_dir,
        #     image_filename=image_filename
        # )
        # all_threads.append(vis_thread)
        
        # Release memory
        del simulator, image
        
        # Limit parallel threads to avoid system overload
        active_threads = [t for t in all_threads if t.is_alive()]
        while len(active_threads) >= max_parallel_processes:
            time.sleep(0.5)
            active_threads = [t for t in all_threads if t.is_alive()]
        
        print(f"Active background threads: {len(active_threads)}/{len(all_threads)}")

        t_end_total = time.time()
        print(f"Elapsed total time: {t_end_total - t_start_total}")
        
    # Wait for all background threads to complete
    print(f"\nWaiting for {len([t for t in all_threads if t.is_alive()])} background threads to complete...")
    for thread in all_threads:
        thread.join()
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
