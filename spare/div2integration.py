import os
import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def merge_sinogram_files(input_base_dir, output_base_dir):
    """
    Merges individual (182, 365) .npy files back into original (182, 365, 1764) format.
    
    Args:
        input_base_dir: Directory containing the split .npy files
        output_base_dir: Directory where the merged files will be saved
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Dictionary to group files by their base name
    file_groups = defaultdict(list)
    
    # Pattern to extract base name and slice index
    pattern = re.compile(r'(.+)_(\d+)\.npy$')
    
    # Find all .npy files and group them
    print("Scanning for .npy files and grouping them...")
    for dirpath, _, filenames in os.walk(input_base_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                match = pattern.match(filename)
                if match:
                    base_name = match.group(1)
                    slice_idx = int(match.group(2))
                    rel_dirpath = os.path.relpath(dirpath, input_base_dir)
                    
                    # Store tuple of (filepath, slice_idx)
                    file_groups[(rel_dirpath, base_name)].append(
                        (os.path.join(dirpath, filename), slice_idx)
                    )
    
    print(f"Found {len(file_groups)} file groups to process")
    
    # Process each group
    for (rel_dirpath, base_name), files in tqdm(file_groups.items(), desc="Merging file groups"):
        # Create output directory
        if rel_dirpath != '.':
            output_dir = os.path.join(output_base_dir, rel_dirpath)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = output_base_dir
        
        # Sort files by slice index
        files.sort(key=lambda x: x[1])
        
        # Check if we have all 1764 slices
        if len(files) != 1764:
            print(f"Warning: Group {base_name} in {rel_dirpath} has {len(files)} slices instead of 1764")
            continue
        
        try:
            # Load first file to get dimensions
            first_slice = np.load(files[0][0])
            if first_slice.shape != (182, 365):
                print(f"Warning: File {files[0][0]} has unexpected shape {first_slice.shape}, expected (182, 365)")
                continue
            
            # Initialize the 3D array
            merged_data = np.zeros((182, 365, 1764), dtype=first_slice.dtype)
            
            # Load and stack all files
            for filepath, slice_idx in files:
                slice_data = np.load(filepath)
                # Convert 1-based index to 0-based for array indexing
                merged_data[:, :, slice_idx-1] = slice_data
            
            # Save merged file
            output_filename = f"{base_name}.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, merged_data)
            print(f"Saved merged file: {output_path}")
            
        except Exception as e:
            print(f"Error processing group {base_name} in {rel_dirpath}: {str(e)}")
    
    print("Processing complete!")

def main():
    # Process both train and test directories
    # D:\fyq\sinogram\log\cv_results\all_prediction
    input_dir = "/mnt/d/fyq/sinogram/log/cv_results/all_prediction"
    output_dir = "/mnt/d/fyq/sinogram/log/cv_results/all_prediction_merged"
    
    if os.path.exists(input_dir):
        print(f"Processing files in {input_dir}...")
        merge_sinogram_files(input_dir, output_dir)
    else:
        print(f"Directory {input_dir} not found, skipping")
    
    print("All processing complete!")

if __name__ == "__main__":
    main()