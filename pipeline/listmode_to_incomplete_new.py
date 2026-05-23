#!/usr/bin/env python3
"""
listmode_to_incomplete_new.py

与 listmode_to_incomplete.py 相同，但适配了新的文件命名格式：
  输入:  listmode_{image_filename}.npz   (如 listmode_002_S_5018.npz)
  完整sinogram: reconstructed_{image_filename}.npy
  输出:  incomplete_index{file_index}_num{num_events}.npy  (file_index 为排序后的 0-based 整数)

改动了三处（均有 # [CHANGED] 标注）：
  1. load_complete_sinogram: 用 image_filename 查找完整 sinogram
  2. process_listmode_file:  移除正则解析，改为接收 file_index + image_filename 参数
  3. main() 循环:            从文件名提取 image_filename，传入 file_index 和 image_filename

Usage:
  python listmode_to_incomplete_new.py \
      --input_dir "E:\\data\\pet_output\\2000000000\\listmode" \
      --output_dir "E:\\data\\pet_output\\2000000000" \
      --num_events 2000000000
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
    def visualize_sinogram(sinogram, output_path, title="Sinogram Visualization", image_filename=None):
        matplotlib.use('Agg')
        try:
            if torch.is_tensor(sinogram):
                sinogram_np = sinogram.cpu().numpy()
            else:
                sinogram_np = sinogram
            sinogram_shape = sinogram_np.shape
            print(f"Sinogram shape: {sinogram_shape}")
            fig, ax = plt.subplots(figsize=(10, 6))
            slice_idx = min(42, sinogram_shape[2])
            im = ax.imshow(sinogram_np[0, :, :slice_idx], cmap='magma')
            if image_filename:
                ax.set_title(f"{title} ({image_filename})")
            else:
                ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            print(f"  -> Saved sinogram visualization to {output_path}")
        except Exception as e:
            print(f"Warning: Failed to create sinogram visualization: {e}")
            import traceback
            traceback.print_exc()

def visualize_sinogram_multislice(sinogram, output_path, title="Sinogram Multislice", num_slices=6, image_filename=None):
    matplotlib.use('Agg')
    try:
        if torch.is_tensor(sinogram):
            sinogram_np = sinogram.cpu().numpy()
        else:
            sinogram_np = sinogram
        sinogram_shape = sinogram_np.shape
        print(f"Generating multislice visualization for sinogram of shape {sinogram_shape}")
        depth = min(42, sinogram_shape[2])
        if num_slices > depth:
            num_slices = depth
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
        rows = (num_slices + 2) // 3
        cols = min(3, num_slices)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([axs])
        else:
            axs = axs.flatten()
        for i, slice_idx in enumerate(slice_indices):
            if i < len(axs):
                im = axs[i].imshow(sinogram_np[:, :, slice_idx], cmap='magma', aspect='auto')
                axs[i].set_title(f'Ring Slice {slice_idx}')
                axs[i].set_xlabel('Radial Position')
                axs[i].set_ylabel('Angle')
        for i in range(num_slices, len(axs)):
            axs[i].set_visible(False)
        if image_filename:
            fig.suptitle(f"{title} ({image_filename})", fontsize=16)
        else:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"  -> Saved multislice sinogram visualization to {output_path}")
    except Exception as e:
        print(f"Warning: Failed to create multislice visualization: {e}")
        import traceback
        traceback.print_exc()

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
    'NrCrystalsPerRing': 364,
    'NrRings': 42,
    'firstCrystalAxis': 0
}

MISSING_SECTORS = [(30, 90), (210, 270)]
DEBUG = True

def build_missing_detector_ids(crystals_per_ring, num_rings, missing_sectors):
    missing_ids = set()
    if DEBUG:
        print("Building missing detector IDs...")
        print(f"  crystals_per_ring = {crystals_per_ring}, num_rings = {num_rings}")
        print(f"  missing_sectors = {missing_sectors}")
    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            angle_deg = (360.0 / crystals_per_ring) * crystal_idx
            is_missing = False
            for (deg_start, deg_end) in missing_sectors:
                if deg_start <= angle_deg <= deg_end:
                    is_missing = True
                    break
            if is_missing:
                det_id = ring_idx * crystals_per_ring + crystal_idx
                missing_ids.add(det_id)
                if DEBUG and len(missing_ids) < 10:
                    print(f"  Debug: ring={ring_idx}, crystal={crystal_idx}, "
                          f"angle={angle_deg:.2f} deg => det_id={det_id} (missing)")
    if DEBUG:
        print(f"Built missing detector set with {len(missing_ids)} detectors "
              f"({(len(missing_ids) / (crystals_per_ring * num_rings) * 100):.1f}% of total).")
    return missing_ids

def visualize_detector_coverage(crystals_per_ring, num_rings, missing_ids, missing_sectors, output_dir):
    angles_active, rings_active, angles_missing, rings_missing = [], [], [], []
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
    for (deg_start, deg_end) in missing_sectors:
        plt.axvspan(deg_start, deg_end, color='red', alpha=0.1)
    plt.xlim(0, 360)
    plt.ylim(0, num_rings)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "detector_coverage.png"), dpi=300)
    plt.close()
    print(f"Saved detector coverage visualization to {output_dir}/detector_coverage.png")

def filter_listmode_data(events, missing_ids):
    if hasattr(events, 'dtype') and events.dtype.names is not None:
        if 'det1_id' in events.dtype.names and 'det2_id' in events.dtype.names:
            det1_ids = events['det1_id']
            det2_ids = events['det2_id']
        else:
            field_names = events.dtype.names
            det1_ids = events[field_names[0]]
            det2_ids = events[field_names[1]]
    elif len(events.shape) >= 2 and events.shape[1] >= 2:
        det1_ids = events[:, 0]
        det2_ids = events[:, 1]
    else:
        raise ValueError(f"Unsupported event data format: shape={events.shape}, dtype={events.dtype}")
    mask_missing_det1 = np.isin(det1_ids, list(missing_ids))
    mask_missing_det2 = np.isin(det2_ids, list(missing_ids))
    valid_mask = ~(mask_missing_det1 | mask_missing_det2)
    if hasattr(events, 'dtype') and events.dtype.names is not None:
        return events[valid_mask]
    elif len(events.shape) >= 2:
        return events[valid_mask]
    else:
        raise ValueError("Unable to filter events: unsupported data format")

# [CHANGED 1] 用 image_filename 直接查找完整 sinogram（原版用整数 index + num_events）
def load_complete_sinogram(sinogram_dir, image_filename):
    """
    Load the complete sinogram corresponding to image_filename.
    Looks for: {sinogram_dir}/reconstructed_{image_filename}.npy
    """
    if sinogram_dir is None:
        return None
    complete_path = os.path.join(sinogram_dir, f"reconstructed_{image_filename}.npy")
    if os.path.exists(complete_path):
        try:
            return np.load(complete_path)
        except Exception as e:
            print(f"Error loading complete sinogram from {complete_path}: {e}")
    else:
        print(f"Complete sinogram not found: {complete_path}")
    return None

def process_and_compare_sinograms_background(complete_sinogram, incomplete_sinogram, output_dir, log_dir, image_index):
    def _process_and_compare():
        matplotlib.use('Agg')
        try:
            vis_dir = os.path.join(log_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            if torch.is_tensor(incomplete_sinogram):
                incomplete_sinogram_np = incomplete_sinogram.cpu().numpy()
            else:
                incomplete_sinogram_np = incomplete_sinogram
            incomplete_shape = incomplete_sinogram_np.shape
            complete_shape = complete_sinogram.shape
            print(f"Sinogram shapes - Complete: {complete_shape}, Incomplete: {incomplete_shape}")
            if complete_shape != incomplete_shape:
                print(f"Warning: Sinogram shapes do not match. Resizing for visualization.")
                min_shape = [min(s1, s2) for s1, s2 in zip(complete_shape, incomplete_shape)]
                incomplete_sinogram_np = incomplete_sinogram_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                complete_sinogram_crop = complete_sinogram[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                complete_sinogram_crop = complete_sinogram
            difference = complete_sinogram_crop - incomplete_sinogram_np
            fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
            slice_idx = min(42, incomplete_shape[2]-1)
            im1 = axs1[0].imshow(complete_sinogram_crop[0, :, :slice_idx], cmap='magma')
            axs1[0].set_title(f'Complete Ring Sinogram (First {slice_idx} Slices)')
            axs1[0].axis('off')
            fig1.colorbar(im1, ax=axs1[0], fraction=0.046, pad=0.04)
            im2 = axs1[1].imshow(incomplete_sinogram_np[0, :, :slice_idx], cmap='magma')
            axs1[1].set_title(f'Incomplete Ring Sinogram (First {slice_idx} Slices)')
            axs1[1].axis('off')
            fig1.colorbar(im2, ax=axs1[1], fraction=0.046, pad=0.04)
            im3 = axs1[2].imshow(difference[0, :, :slice_idx], cmap='coolwarm')
            axs1[2].set_title(f'Difference (Complete - Incomplete)')
            axs1[2].axis('off')
            fig1.colorbar(im3, ax=axs1[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig1_filename = os.path.join(vis_dir, f"sinogram_comparison_index{image_index}.pdf")
            plt.savefig(fig1_filename, dpi=300)
            plt.close(fig1)
            print(f"  -> Saved sinogram comparison to {fig1_filename}")
        except Exception as e:
            print(f"Warning: Failed to generate sinogram comparison: {e}")
            import traceback
            traceback.print_exc()
    thread = threading.Thread(target=_process_and_compare)
    thread.daemon = False
    thread.start()
    return thread

# [CHANGED 2] 新增 file_index 和 image_filename 参数，移除内部正则解析
def process_listmode_file(input_file, output_dir, complete_sinogram_dir, log_dir,
                          missing_ids, num_events, file_index, image_filename, vis_level=2):
    """
    Process a single listmode file: filter it and generate a sinogram.

    Args:
        input_file:           Path to the input listmode .npz file
        output_dir:           Base output directory
        complete_sinogram_dir: Directory containing complete sinograms for comparison
        log_dir:              Directory for visualizations and logs
        missing_ids:          Set of detector IDs considered missing
        num_events:           Event count for output path construction
        file_index:           0-based integer index from the sorted file list  # [CHANGED]
        image_filename:       Stem of the original image (e.g. '002_S_5018')   # [CHANGED]
        vis_level:            Visualization detail level (0=minimal, 1=basic, 2=detailed)
    """
    # [CHANGED] 不再从文件名解析 index，直接使用传入的 file_index 和 image_filename
    index = file_index
    print(f"\nProcessing listmode data for index {index} (image: {image_filename})...")

    start_time = time.time()

    # Load listmode data
    try:
        data = np.load(input_file)
        if isinstance(data, np.ndarray):
            events_data = data
        else:
            if 'listmode' in data:
                events_data = data['listmode']
            else:
                try:
                    events_data = next(iter(data.values()))
                except:
                    print(f"Data keys: {list(data.keys())}")
                    raise ValueError(f"Cannot extract event data from {input_file}")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return

    print(f"Loaded events from {input_file}, format: shape={events_data.shape}, dtype={events_data.dtype}")

    if hasattr(events_data, 'shape'):
        if len(events_data.shape) == 1 and hasattr(events_data, 'dtype') and events_data.dtype.names is not None:
            num_loaded_events = len(events_data)
        else:
            num_loaded_events = events_data.shape[0]
    else:
        num_loaded_events = "unknown"

    print(f"Loaded {num_loaded_events} events from {input_file}")

    # Filter events
    try:
        filtered_events = filter_listmode_data(events_data, missing_ids)
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

    try:
        if hasattr(filtered_events, 'dtype') and filtered_events.dtype.names is not None:
            if 'det1_id' in filtered_events.dtype.names and 'det2_id' in filtered_events.dtype.names:
                detector_ids = torch.tensor(
                    np.column_stack((filtered_events['det1_id'], filtered_events['det2_id'])),
                    dtype=torch.int32
                )
            else:
                field_names = filtered_events.dtype.names
                detector_ids = torch.tensor(
                    np.column_stack((filtered_events[field_names[0]], filtered_events[field_names[1]])),
                    dtype=torch.int32
                )
        else:
            detector_ids = torch.from_numpy(filtered_events[:, :2]).to(torch.int32)

        print("Generating sinogram from incomplete data...")
        incomplete_sinogram = gate.listmode_to_sinogram(detector_ids, info)

        out_sinogram_path = os.path.join(incomplete_sinogram_dir, f"incomplete_index{index}_num{num_events}")
        np.save(out_sinogram_path, incomplete_sinogram.numpy().astype(np.float32))
        print(f"Saved incomplete sinogram to {out_sinogram_path}.npy")

        if vis_level >= 1:
            standard_vis_path = os.path.join(vis_dir, f"sinogram_{index}.pdf")
            visualize_sinogram(
                sinogram=incomplete_sinogram,
                output_path=standard_vis_path,
                title="Incomplete Ring Sinogram",
                image_filename=image_filename
            )
            multislice_vis_path = os.path.join(vis_dir, f"sinogram_multislice_{image_filename}.pdf")
            visualize_sinogram_multislice(
                sinogram=incomplete_sinogram,
                output_path=multislice_vis_path,
                title="Incomplete Ring Sinogram Multislice",
                num_slices=6,
                image_filename=image_filename
            )

        # [CHANGED] 用 image_filename 查找对应的完整 sinogram
        complete_sinogram = load_complete_sinogram(complete_sinogram_dir, image_filename)

        if complete_sinogram is not None and vis_level >= 2:
            print("Found matching complete sinogram, generating comparison...")
            process_and_compare_sinograms_background(
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

    save_thread.join()
    print(f"Completed processing index {index} in {time.time() - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Convert complete listmode data to incomplete ring data (new filename format)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing listmode_{image_filename}.npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--sinogram_dir', type=str,
                        help='Directory containing reconstructed_{image_filename}.npy (complete sinograms)')
    parser.add_argument('--num_events', type=int, required=True,
                        help='Number of events (used for output filename)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate detector coverage visualization')
    parser.add_argument('--vis_level', type=int, default=2, choices=[0, 1, 2],
                        help='Visualization detail level (0=minimal, 1=basic, 2=detailed)')
    parser.add_argument('--missing_start1', type=float, default=30)
    parser.add_argument('--missing_end1',   type=float, default=90)
    parser.add_argument('--missing_start2', type=float, default=210)
    parser.add_argument('--missing_end2',   type=float, default=270)
    parser.add_argument('--missing_start3', type=float, default=0)
    parser.add_argument('--missing_end3',   type=float, default=0)
    parser.add_argument('--missing_start4', type=float, default=0)
    parser.add_argument('--missing_end4',   type=float, default=0)

    args = parser.parse_args()

    if args.missing_start3:
        missing_sectors = [
            (args.missing_start1, args.missing_end1),
            (args.missing_start2, args.missing_end2),
            (args.missing_start3, args.missing_end3),
            (args.missing_start4, args.missing_end4),
        ]
    else:
        missing_sectors = [
            (args.missing_start1, args.missing_end1),
            (args.missing_start2, args.missing_end2),
        ]

    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join(args.output_dir, 'log_incomplete', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Resolve sinogram_dir
    sinogram_dir = args.sinogram_dir
    if not sinogram_dir:
        parent_dir = os.path.dirname(args.input_dir)
        sinogram_dir = os.path.join(parent_dir, 'sinogram')
        if not os.path.exists(sinogram_dir):
            print(f"Complete sinogram directory not found at {sinogram_dir}")
            print("Visualizations will not include comparisons")
            sinogram_dir = None

    missing_ids = build_missing_detector_ids(
        crystals_per_ring=info['NrCrystalsPerRing'],
        num_rings=info['NrRings'],
        missing_sectors=missing_sectors
    )

    if args.visualize:
        visualize_detector_coverage(
            crystals_per_ring=info['NrCrystalsPerRing'],
            num_rings=info['NrRings'],
            missing_ids=missing_ids,
            missing_sectors=missing_sectors,
            output_dir=log_dir
        )

    # Find and sort all listmode files
    if os.path.isdir(args.input_dir):
        listmode_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    else:
        listmode_files = [args.input_dir] if os.path.exists(args.input_dir) else []

    print(f"Found {len(listmode_files)} listmode files to process")
    if not listmode_files:
        print("No input files found. Check the --input_dir path.")
        return

    # [CHANGED 3] 从文件名提取 image_filename，传入 file_index（0-based）
    for i, lm_file in enumerate(listmode_files):
        basename = os.path.basename(lm_file)          # e.g. listmode_002_S_5018.npz
        # Strip "listmode_" prefix and ".npz" suffix
        if basename.startswith("listmode_") and basename.endswith(".npz"):
            image_filename = basename[len("listmode_"):-len(".npz")]  # e.g. 002_S_5018
        else:
            image_filename = os.path.splitext(basename)[0]            # fallback

        print(f"\n[{i+1}/{len(listmode_files)}] Processing {lm_file}  (index={i}, image={image_filename})")

        process_listmode_file(
            input_file=lm_file,
            output_dir=args.output_dir,
            complete_sinogram_dir=sinogram_dir,
            log_dir=log_dir,
            missing_ids=missing_ids,
            num_events=args.num_events,
            file_index=i,           # [CHANGED] 0-based 整数索引
            image_filename=image_filename,  # [CHANGED] 原始图像文件名 stem
            vis_level=args.vis_level
        )

    print("\nAll files processed. Incomplete ring data generation complete.")
    print(f"Incomplete sinograms saved to: {os.path.join(args.output_dir, 'sinogram_incomplete')}")
    print(f"  Named as: incomplete_index{{0..{len(listmode_files)-1}}}_num{args.num_events}.npy")

if __name__ == "__main__":
    main()
