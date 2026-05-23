import os
import numpy as np
from tqdm import tqdm

def pad_npy_files(input_dir, output_dir=None, overwrite=False):
    """
    Pad .npy files from shape (80, 128, 128) to (128, 128, 128) by adding zeros.
    
    Args:
        input_dir: Directory containing the .npy files
        output_dir: Directory to save the padded files (if None and not overwriting, will create a "padded" subfolder)
        overwrite: If True, overwrite the original files; if False, create new files in output_dir
    """
    # Set up output directory
    if overwrite:
        output_dir = input_dir
    elif output_dir is None:
        # Create a "padded" subfolder in the input directory
        output_dir = os.path.join(input_dir, "padded")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} .npy files in {input_dir}")
    
    # Process each file
    for filename in tqdm(npy_files, desc="Padding files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load the array
        array = np.load(input_path)
        
        # Check shape
        if array.shape == (80, 128, 128):
            # Calculate padding - add 24 zeros before and 24 zeros after first dimension
            # to evenly center the original data
            pad_width = ((24, 24), (0, 0), (0, 0))
            
            # Pad the array
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=1e-8)
            
            # Save the padded array
            np.save(output_path, padded_array)
            print(f"Padded {filename} from {array.shape} to {padded_array.shape}")
        else:
            print(f"Warning: {filename} has shape {array.shape}, not (80, 128, 128). Skipping padding.")
            # If not overwriting, copy the original file to the output directory
            if not overwrite:
                np.save(output_path, array)
    
    print(f"Processing complete. Padded files saved to {output_dir}")

# Set the input directory
input_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm"

# Specify the output directory or set to None to use a "padded" subfolder
output_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8"  # You can change this to a specific path if needed

# Set overwrite to False to create new files (safer option)
overwrite = False

# Run the function
pad_npy_files(input_dir, output_dir, overwrite)