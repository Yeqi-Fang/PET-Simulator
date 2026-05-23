import numpy as np
import os
from scipy.io import savemat
import re

def convert_npy_to_mat():
    """
    Convert .npy files from two folders back to the original .mat format.
    
    Each .npy file has shape (128, 128, 128) and will be converted into 128 .mat files,
    each with shape [1, 256, 128].
    """
    # Define the paths to the two folders
    folder1 = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8_rotated"
    folder2 = r'C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8'

    # Create an output folder for .mat files if it doesn't exist
    output_folder = r'C:\Users\fangy\Desktop\converted_mat_files_rm_1e-8'
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all .npy files in the first folder
    npy_files = [f for f in os.listdir(folder1) if f.endswith('.npy')]
    total_files = len(npy_files)
    
    print(f"Found {total_files} .npy files to process")
    
    processed_files = 0
    skipped_files = 0
    
    for npy_file in npy_files:
        try:
            # Extract information from the filename
            if 'train' in npy_file:
                prefix = 'train'
                # Use regex to extract the number
                match = re.search(r'train_incomplete_(\d+)\.npy', npy_file)
                if match:
                    i = match.group(1)
                else:
                    print(f"Could not parse filename: {npy_file}")
                    skipped_files += 1
                    continue
            elif 'test' in npy_file:
                prefix = 'test'
                # Use regex to extract the number
                match = re.search(r'test_incomplete_(\d+)\.npy', npy_file)
                if match:
                    i = match.group(1)
                else:
                    print(f"Could not parse filename: {npy_file}")
                    skipped_files += 1
                    continue
            else:
                print(f"Skipping unknown file type: {npy_file}")
                skipped_files += 1
                continue
            
            # Check if the corresponding file exists in folder2
            if npy_file not in os.listdir(folder2):
                print(f"Warning: {npy_file} not found in second folder. Skipping.")
                skipped_files += 1
                continue
            
            # Load the two .npy files (first part and second part)
            vol1 = np.load(os.path.join(folder1, npy_file))
            vol2 = np.load(os.path.join(folder2, npy_file))
            vol1 = np.flip(vol1, axis=2)
            # vol1 = np.flip(vol1, axis=1)
            
            # Ensure the volumes have the expected shape
            if vol1.shape != (128, 128, 128) or vol2.shape != (128, 128, 128):
                print(f"Unexpected shape for {npy_file}: {vol1.shape}, {vol2.shape}. Skipping.")
                skipped_files += 1
                continue
            
            # For each slice in the z-direction, create a .mat file
            for j in range(128):
                # Extract the j-th slice from both volumes
                slice1 = vol1[j, :, :]
                slice2 = vol2[j, :, :]
                
                # Create a combined array with shape [1, 256, 128]
                combined = np.zeros((1, 256, 128))
                combined[0, 0:128, :] = slice1
                combined[0, 128:256, :] = slice2
                
                # Save as .mat file
                mat_filename = f"{prefix}_{i}_{j}.mat"
                savemat(os.path.join(output_folder, mat_filename), {'img': combined})
            
            processed_files += 1
            print(f"Processed {npy_file} ({processed_files}/{total_files})")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            skipped_files += 1
    
    print(f"Conversion completed! Processed {processed_files} files, skipped {skipped_files} files.")

if __name__ == "__main__":
    convert_npy_to_mat()