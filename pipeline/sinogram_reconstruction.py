#!/usr/bin/env python3
"""
sinogram_reconstruction.py

This script reconstructs 3D volumes from sinogram data (.npy files).
It performs the following steps:
  1) Loads sinogram data from .npy files
  2) Reconstructs using OSEM algorithm with optional PSF correction
  3) Applies optional outlier removal & normalization
  4) Saves reconstructed volumes as .npy files

Usage:
  python sinogram_reconstruction.py --sinogram_dir /path/to/sinograms --output_dir /path/to/output
"""

import os
import re
import glob
import numpy as np
import torch
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.algorithms import OSEM, BSREM
from pytomography.priors import RelativeDifferencePrior
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter

# Import outlier detection functions if needed
try:
    from outlier_detection import (
        global_outlier_detection,
        local_outlier_detection,
        edge_outlier_detection,
        combined_outlier_detection,
        analyze_outlier_masks,
        remove_outliers_iteratively
    )
    OUTLIER_AVAILABLE = True
except ImportError:
    print("Warning: outlier_detection module not found. Outlier removal will be disabled.")
    OUTLIER_AVAILABLE = False

# PET scanner configuration (from sinogram.py)
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

def reconstruct_volume_from_sinogram(
        sinogram_file=None,
        sinogram_data=None,
        voxel_size=2.78,
        volume_size=128,
        n_iters=2,
        n_subsets=24,
        psf_fwhm_mm=1.0,
        use_psf=True,
        use_prior=False,
        beta=25,
        gamma=2,
        apply_outlier_removal=False) -> np.ndarray:
    """
    Reconstruct a 3D volume from sinogram data.
    
    Args:
        sinogram_file (str, optional): Path to the sinogram .npy file.
        sinogram_data (np.ndarray, optional): Directly provided sinogram data.
        voxel_size (float): The voxel size in mm.
        volume_size (int): The volume dimension (cube).
        n_iters (int): Number of iterations.
        n_subsets (int): Number of subsets for OSEM/BSREM.
        psf_fwhm_mm (float): If >0 and use_psf=True, apply a Gaussian PSF with given FWHM in mm.
        use_psf (bool): Whether to apply point spread function correction.
        use_prior (bool): Whether to use a prior (BSREM) instead of OSEM.
        beta (float): Beta parameter for RelativeDifferencePrior (if use_prior=True).
        gamma (float): Gamma parameter for RelativeDifferencePrior (if use_prior=True).
        apply_outlier_removal (bool): Whether to apply outlier detection and removal.

    Returns:
        np.ndarray: Reconstructed 3D volume after processing and normalization.
    """
    # Get sinogram data either from provided array or from file
    if sinogram_data is not None:
        sinogram = sinogram_data
    elif sinogram_file is not None:
        # Load from file
        sinogram = np.load(sinogram_file)
    else:
        raise ValueError("Either sinogram_file or sinogram_data must be provided")
    
    # Convert numpy array to torch tensor if needed
    if isinstance(sinogram, np.ndarray):
        sinogram = torch.from_numpy(sinogram)
    
    print(f"Sinogram shape: {sinogram.shape}")
    
    # Define object space reconstruction matrix
    object_meta = ObjectMeta(
        dr=(voxel_size, voxel_size, voxel_size),  # mm
        shape=(128, 128, 80)  # voxels
    )
    
    # Define projection space metadata
    proj_meta = PETSinogramPolygonProjMeta(info)
    
    # Set up PSF correction if requested
    obj2obj_transforms = []
    if use_psf and psf_fwhm_mm > 0:
        psf_transform = GaussianFilter(psf_fwhm_mm)
        obj2obj_transforms.append(psf_transform)
    
    # Create system matrix
    system_matrix = PETSinogramSystemMatrix(
        object_meta,
        proj_meta,
        obj2obj_transforms=obj2obj_transforms,
        sinogram_sensitivity=None,
        N_splits=10,  # Split FP/BP into 10 loops to save memory
        attenuation_map=None,
        device='cpu'  # projections on CPU, internal computation on GPU
    )
    
    # Set up likelihood and reconstruction algorithm
    likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,  # Using the sinogram directly as data
    )
    
    # Choose between OSEM and BSREM based on parameters
    if use_prior:
        prior = RelativeDifferencePrior(beta=beta, gamma=gamma)
        recon_algorithm = BSREM(likelihood, prior=prior)
    else:
        recon_algorithm = OSEM(likelihood)
    
    # Reconstruct
    print(f"Starting reconstruction with {n_iters} iterations and {n_subsets} subsets...")
    recon_image = recon_algorithm(n_iters=n_iters, n_subsets=n_subsets)
    
    # Convert to NumPy for further processing
    recon_np = recon_image.cpu().numpy()
    
    # Just normalize without outlier removal
    min_val = recon_np.min()
    max_val = recon_np.max()
    if max_val - min_val > 1e-9:
        final_image = (recon_np - min_val) / (max_val - min_val)
    else:
        final_image = recon_np.copy()
    
    return final_image

def create_visualization(image, output_file=None, slice_index=None):
    """
    Create a visualization of the reconstructed image.
    
    Args:
        image: 3D volume to visualize
        output_file: Path to save the visualization (if None, display only)
        slice_index: Index of the slice to visualize (if None, use middle slice)
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image[:, :, slice_index], cmap='magma', interpolation='gaussian')
    plt.axis('off')
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image (Slice {slice_index})')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to process sinogram files and reconstruct volumes.
    """
    parser = argparse.ArgumentParser(description='Reconstruct volumes from sinogram data')
    parser.add_argument('--sinogram_dir', type=str, required=True, 
                        help='Directory containing sinogram .npy files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save reconstructed volumes')
    parser.add_argument('--pattern', type=str, default='*.npy',
                        help='File pattern for sinogram files (default: *.npy)')
    parser.add_argument('--voxel_size', type=float, default=2.78,
                        help='Voxel size in mm (default: 2.78)')
    parser.add_argument('--volume_size', type=int, default=128,
                        help='Output volume size in voxels (default: 128)')
    parser.add_argument('--n_iters', type=int, default=2,
                        help='Number of iterations (default: 2)')
    parser.add_argument('--n_subsets', type=int, default=24,
                        help='Number of subsets (default: 24)')
    parser.add_argument('--use_psf', action='store_true',
                        help='Apply point spread function correction')
    parser.add_argument('--psf_fwhm_mm', type=float, default=1.0,
                        help='PSF FWHM in mm (default: 1.0)')
    parser.add_argument('--use_prior', action='store_true',
                        help='Use RelativeDifferencePrior (BSREM) instead of OSEM')
    parser.add_argument('--beta', type=float, default=25,
                        help='Beta parameter for RelativeDifferencePrior (default: 25)')
    parser.add_argument('--gamma', type=float, default=2,
                        help='Gamma parameter for RelativeDifferencePrior (default: 2)')
    parser.add_argument('--outlier_removal', action='store_true',
                        help='Apply outlier detection and removal')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of reconstructed volumes')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='Directory to save visualizations (default: output_dir/vis)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = args.vis_dir if args.vis_dir else os.path.join(args.output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Find all sinogram files
    pattern = os.path.join(args.sinogram_dir, args.pattern)
    sinogram_files = sorted(glob.glob(pattern))
    
    if not sinogram_files:
        print(f"No sinogram files found matching pattern {pattern}")
        return
    
    print(f"Found {len(sinogram_files)} sinogram files to process")
    
    # Process each file
    for i, sinogram_file in enumerate(tqdm(sinogram_files, desc="Processing sinograms")):
        print(f"\n[{i+1}/{len(sinogram_files)}] Processing {sinogram_file}")
        start_time = time.time()
        
        # Get base filename
        base_name = os.path.basename(sinogram_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        
        # Try to extract index using regex if filename follows a pattern
        # Adjust this regex as needed based on your filename format
        match = re.search(r'(?:reconstructed|incomplete)_index(\d+)', file_name_no_ext)
        if match:
            index = match.group(1)
            output_name = f"reconstructed_from_sinogram_index{index}.npy"
        else:
            output_name = f"reconstructed_from_sinogram_{file_name_no_ext}.npy"
        
        # Full output path
        output_path = os.path.join(args.output_dir, output_name)
        
        # Skip if already exists (optional - remove if you want to reprocess)
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping.")
            continue
        
        # Reconstruct
        result_3d = reconstruct_volume_from_sinogram(
            sinogram_file=sinogram_file,
            voxel_size=args.voxel_size,
            volume_size=args.volume_size,
            n_iters=args.n_iters,
            n_subsets=args.n_subsets,
            psf_fwhm_mm=args.psf_fwhm_mm,
            use_psf=args.use_psf,
            use_prior=args.use_prior,
            beta=args.beta,
            gamma=args.gamma,
            apply_outlier_removal=args.outlier_removal
        )
        
        # Save result
        np.save(output_path, result_3d.transpose(2, 1, 0))
        
        # Create visualization if requested
        if args.visualize:
            vis_path = os.path.join(vis_dir, f"{file_name_no_ext}_visualization.png")
            create_visualization(result_3d, output_file=vis_path)
        
        elapsed_time = time.time() - start_time
        print(f"  -> Saved {output_path} (shape={result_3d.shape}) in {elapsed_time:.1f}s")
        
    
    print("\nAll reconstructions complete.")

if __name__ == "__main__":
    main()
    
    
# python sinogram_reconstruction.py --sinogram_dir /path/to/sinograms --output_dir /path/to/output --visualize