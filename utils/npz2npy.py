#!/usr/bin/env python3
"""
npz_to_npy_converter.py

A script to convert all .npz files in a specified directory to .npy files.
This is useful for converting compressed NumPy files to uncompressed ones
for faster loading or when working with libraries that only support .npy format.

Usage:
  python npz_to_npy_converter.py --input_dir /path/to/npz/files --output_dir /path/to/output
"""

import os
import glob
import numpy as np
import argparse
import time
from tqdm import tqdm
import threading
import multiprocessing

def convert_npz_to_npy(npz_file, output_dir, array_name=None, verbose=True):
    """
    Convert a .npz file to a .npy file
    
    Args:
        npz_file: Path to the .npz file
        output_dir: Directory to save the .npy file
        array_name: Name of the array to extract (if None, uses the first array)
        verbose: Whether to print progress information
    
    Returns:
        Path to the created .npy file
    """
    try:
        # Get the base filename without extension
        base_name = os.path.basename(npz_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        
        # Load the npz file
        with np.load(npz_file) as data:
            # If array_name is not provided, use the first array
            if array_name is None:
                if len(data.files) == 0:
                    if verbose:
                        print(f"Warning: {npz_file} contains no arrays. Skipping.")
                    return None
                array_name = data.files[0]
                if verbose:
                    print(f"Using first array '{array_name}' from {npz_file}")
            
            # Check if the specified array exists
            if array_name not in data.files:
                if verbose:
                    print(f"Warning: Array '{array_name}' not found in {npz_file}. Available arrays: {data.files}")
                return None
            
            # Extract the array
            array_data = data[array_name]
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{file_name_no_ext}.npy")
            
            # Save as .npy
            np.save(output_file, array_data)
            
            if verbose:
                print(f"Converted {npz_file} -> {output_file} (shape: {array_data.shape}, dtype: {array_data.dtype})")
            
            return output_file
    
    except Exception as e:
        if verbose:
            print(f"Error converting {npz_file}: {str(e)}")
        return None

def process_file(args):
    """Function for multiprocessing pool to convert a single file"""
    npz_file, output_dir, array_name, verbose = args
    return convert_npz_to_npy(npz_file, output_dir, array_name, verbose)

def convert_directory(input_dir, output_dir, array_name=None, use_multiprocessing=True, num_processes=None, verbose=True):
    """
    Convert all .npz files in a directory to .npy files
    
    Args:
        input_dir: Directory containing .npz files
        output_dir: Directory to save .npy files
        array_name: Name of the array to extract from each .npz file
        use_multiprocessing: Whether to use multiprocessing for parallel conversion
        num_processes: Number of processes to use (default: CPU count)
        verbose: Whether to print progress information
    
    Returns:
        List of created .npy files
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npz files
    npz_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return []
    
    print(f"Found {len(npz_files)} .npz files in {input_dir}")
    
    if use_multiprocessing and len(npz_files) > 1:
        if num_processes is None:
            num_processes = min(multiprocessing.cpu_count(), 8)  # Limit to 8 processes by default
        
        print(f"Using {num_processes} processes for parallel conversion")
        
        # Prepare arguments for each file
        args_list = [(npz_file, output_dir, array_name, False) for npz_file in npz_files]
        
        # Convert files in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, args_list), total=len(npz_files), desc="Converting files"))
        
        # Filter out None results (failed conversions)
        output_files = [f for f in results if f is not None]
    else:
        # Convert files sequentially
        output_files = []
        for npz_file in tqdm(npz_files, desc="Converting files"):
            output_file = convert_npz_to_npy(npz_file, output_dir, array_name, verbose)
            if output_file is not None:
                output_files.append(output_file)
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Convert .npz files to .npy files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--output_dir", type=str, help="Directory to save .npy files (default: input_dir + '_npy')")
    parser.add_argument("--array_name", type=str, help="Name of the array to extract from each .npz file (default: first array)")
    parser.add_argument("--sequential", action="store_true", help="Disable multiprocessing and convert files sequentially")
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing (default: CPU count)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    # If output_dir is not specified, create a '_npy' version of input_dir
    if args.output_dir is None:
        args.output_dir = args.input_dir + '_npy'
    
    start_time = time.time()
    
    # Convert all files
    output_files = convert_directory(
        args.input_dir,
        args.output_dir,
        array_name=args.array_name,
        use_multiprocessing=not args.sequential,
        num_processes=args.processes,
        verbose=not args.quiet
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nConversion complete!")
    print(f"Successfully converted {len(output_files)} files in {elapsed_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()