import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def crop_npy_files(index, input_dir, output_dir, target_shape=(80, 128, 128), crop_strategy='center'):
    """
    Crops NumPy arrays to target_shape and saves them to output_dir.
    
    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save cropped files
        target_shape: Target shape after cropping (default: (80, 128, 128))
        crop_strategy: How to crop the array ('start', 'center', or 'end')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the input directory
    npy_files = list(Path(input_dir).glob('**/*.npy'))
    
    print(f"Found {len(npy_files)} .npy files in {input_dir}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each file
    for file_path in tqdm(npy_files, desc="Processing files"):
        try:
            # Load the NumPy array
            arr = np.load(file_path)
            if index == 1:
                arr = np.flip(arr, axis=2)
            # Check if the array has the right dimensions
            if len(arr.shape) == 3 and arr.shape[0] >= target_shape[0] and arr.shape[1] == target_shape[1] and arr.shape[2] == target_shape[2]:
                # Crop the array to the target shape based on the strategy
                if crop_strategy == 'start':
                    # Take the first elements
                    arr_cropped = arr[:target_shape[0], :, :]
                elif crop_strategy == 'center':
                    # Take the center elements
                    start_idx = (arr.shape[0] - target_shape[0]) // 2
                    end_idx = start_idx + target_shape[0]
                    arr_cropped = arr[start_idx:end_idx, :, :]
                elif crop_strategy == 'end':
                    # Take the last elements
                    arr_cropped = arr[-target_shape[0]:, :, :]
                else:
                    raise ValueError(f"Unknown crop strategy: {crop_strategy}")
                
                # Create the output file path with the same relative path
                rel_path = file_path.relative_to(input_dir)
                output_path = Path(output_dir) / rel_path
                
                # Create parent directories if needed
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Save the cropped array
                np.save(output_path, arr_cropped)
                processed_count += 1
                
                # Print details for the first few files
                if processed_count <= 3:
                    print(f"Example: {file_path.name} -> Shape: {arr.shape} -> {arr_cropped.shape}")
            else:
                print(f"Skipped: {file_path} - Shape {arr.shape} cannot be cropped to {target_shape}")
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            error_count += 1
    
    print(f"Summary for {input_dir}:")
    print(f"  - Processed: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Errors: {error_count}")
    return processed_count, skipped_count, error_count

# Define input and output directories
input_dirs = [
    r"C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8",
    r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8_rotated"
]

output_dirs = [
    r"C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8_80",
    r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8_rotated_80"
]

# Choose the cropping strategy ('start', 'center', or 'end')
# - 'start': Take the first 80 elements along the first dimension
# - 'center': Take the center 80 elements (default)
# - 'end': Take the last 80 elements
crop_strategy = 'center'  # Using 'start' to remove the excess pixels at the end

# Process each input-output directory pair
total_processed = 0
total_skipped = 0
total_errors = 0

for index, (input_dir, output_dir) in enumerate(zip(input_dirs, output_dirs)):
    print(f"\nProcessing directory: {input_dir}")
    processed, skipped, errors = crop_npy_files(index, input_dir, output_dir, crop_strategy=crop_strategy)
    total_processed += processed
    total_skipped += skipped
    total_errors += errors

print("\nOverall Summary:")
print(f"  - Total Processed: {total_processed}")
print(f"  - Total Skipped: {total_skipped}")
print(f"  - Total Errors: {total_errors}")
print(f"  - Crop Strategy: '{crop_strategy}' (keeping the first 80 elements)")
print("\nAll processing complete!")