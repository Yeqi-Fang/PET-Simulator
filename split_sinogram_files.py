import os
import numpy as np
from tqdm import tqdm

def split_sinogram_files(base_dir, output_base_dir):
    """
    Splits sinogram .npy files along their third dimension.
    
    Args:
        base_dir: Directory containing the source .npy files
        output_base_dir: Directory where the split files will be saved
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
    
    # Process each .npy file with a progress bar
    for filepath in tqdm(all_npy_files, desc=f"Processing {base_dir}"):
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
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                
                # Split and save each slice
                for i in range(data.shape[2]):
                    # Extract the slice
                    slice_data = data[:, :, i]
                    
                    # Convert to float16 to reduce file size
                    slice_data = slice_data.astype(np.float16)
                    
                    # Create output filename
                    output_filename = f"{base_filename}_{i+1}.npy"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save the slice
                    np.save(output_path, slice_data)
            else:
                print(f"Skipped: {filepath} - unexpected shape {data.shape}")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

def main():
    # Process both train and test directories
    base_dirs = [r"/root/autodl-tmp/pet_output/2000000000/2e9smooth_MCI/train",
                r"/root/autodl-tmp/pet_output/2000000000/2e9smooth_MCI/test"]
    output_base_dir = r"/root/autodl-tmp/pet_output/2000000000/2e9div_smooth_MCI"
    
    for base_dir in base_dirs:
        # Get the subfolder (train or test)
        subfolder = os.path.basename(base_dir)
        # Set the output directory to maintain structure
        output_dir = os.path.join(output_base_dir, subfolder)
        
        print(f"Processing files in {base_dir}...")
        split_sinogram_files(base_dir, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()