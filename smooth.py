import os
import numpy as np
import torch
from tqdm import tqdm
from pytomography.io.PET import gate
import sys


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


def smooth_sinogram_files(base_dir, output_base_dir, info, batch_size=1):
    """
    Smooths sinogram .npy files using gate.smooth_randoms_sinogram.
    
    Args:
        base_dir: Directory containing the source .npy files
        output_base_dir: Directory where the smoothed files will be saved
        info: PET scanner information object
        batch_size: Number of files to process in a batch (for memory efficiency)
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all .npy files in the directory tree
    all_npy_files = []
    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                all_npy_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(all_npy_files)} .npy files to process")
    
    # Process files in batches to manage memory
    for i in range(0, len(all_npy_files), batch_size):
        batch_files = all_npy_files[i:i+batch_size]
        
        # Process each .npy file in the current batch
        for filepath in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}/{(len(all_npy_files)-1)//batch_size + 1}"):
            try:
                # Get relative path to maintain directory structure
                rel_dirpath = os.path.relpath(os.path.dirname(filepath), base_dir)
                
                # Create corresponding output directory
                if rel_dirpath != '.':
                    output_dir = os.path.join(output_base_dir, rel_dirpath)
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = output_base_dir
                
                # Load the data
                data = np.load(filepath)
                
                # Check if the shape matches what we expect
                if len(data.shape) == 3 and data.shape[2] == 1764:
                    # Convert to torch tensor
                    data_tensor = torch.tensor(data)
                    
                    # Apply smoothing
                    smoothed_data = gate.smooth_randoms_sinogram(
                        data_tensor, 
                        info, 
                        sigma_r=0.7, 
                        sigma_theta=0.7, 
                        sigma_z=0.7, 
                        kernel_size_r=5, 
                        kernel_size_theta=5, 
                        kernel_size_z=5
                    )
                    
                    # Convert back to numpy array
                    smoothed_data_np = smoothed_data.cpu().numpy()
                    
                    # Get the base filename
                    base_filename = os.path.basename(filepath)
                    
                    # Create output filepath
                    output_path = os.path.join(output_dir, base_filename)
                    
                    # Save the smoothed data
                    np.save(output_path, smoothed_data_np)
                else:
                    print(f"Skipped: {filepath} - unexpected shape {data.shape}")
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                import traceback
                traceback.print_exc()

def main():
    # Paths
    base_dirs = [r"D:\data\pet_output\2000000000\2e9_add\train",
                r"D:\data\pet_output\2000000000\2e9_add\test"]
    output_base_dir = r"D:\data\pet_output\2000000000\2e9smooth_add"
    
    # Process both train and test directories
    for base_dir in base_dirs:
        # Get the subfolder (train or test)
        subfolder = os.path.basename(base_dir)
        # Set the output directory to maintain structure
        output_dir = os.path.join(output_base_dir, subfolder)
        
        print(f"Processing files in {base_dir}...")
        smooth_sinogram_files(base_dir, output_dir, info, batch_size=1)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
    
# conda create -n pytomography_env python=3.9
# conda activate pytomography_env

# pip install pytomography tqdm
# conda install -c conda-forge libparallelproj parallelproj cupy

