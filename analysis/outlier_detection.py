import numpy as np
import numba


# 1. Global Outlier Detection
def global_outlier_detection(image: np.ndarray,
                             factor: float = 3.0,
                             percentile: float = 95) -> np.ndarray:
    """
    Global outlier detection: any voxel above factor * (percentile) is flagged.
    """
    flattened = image.ravel()
    ref_value = np.percentile(flattened, percentile)
    threshold = factor * ref_value
    return image > threshold


# 2. Local Outlier Detection (Numba)
@numba.njit
def local_outlier_detection_nb(image: np.ndarray,
                               window_size: int = 3,
                               sigma_factor: float = 3.0) -> np.ndarray:
    """
    Numba-accelerated local outlier detection.
    Returns a uint8 array (1 => outlier, 0 => normal).
    """
    zdim, ydim, xdim = image.shape
    outliers = np.zeros((zdim, ydim, xdim), dtype=np.uint8)  # Not bool!
    half_w = window_size // 2
    
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                val = image[z, y, x]
                
                z1 = max(0, z - half_w)
                z2 = min(zdim, z + half_w + 1)
                y1 = max(0, y - half_w)
                y2 = min(ydim, y + half_w + 1)
                x1 = max(0, x - half_w)
                x2 = min(xdim, x + half_w + 1)
                
                # Manually compute local mean and std dev:
                local_sum = 0.0
                local_sum_sq = 0.0
                count = 0
                
                for zz in range(z1, z2):
                    for yy in range(y1, y2):
                        for xx in range(x1, x2):
                            v = image[zz, yy, xx]
                            local_sum += v
                            local_sum_sq += v * v
                            count += 1

                mean_val = local_sum / count
                var_val = (local_sum_sq / count) - (mean_val * mean_val)
                
                # Avoid false positives in nearly uniform regions
                if var_val < 1e-9:
                    continue
                
                std_val = np.sqrt(var_val)
                
                # If val > mean + sigma_factor*std, mark as outlier
                if val > mean_val + sigma_factor * std_val:
                    outliers[z, y, x] = 1
    
    return outliers

def local_outlier_detection(image: np.ndarray,
                            window_size=3,
                            sigma_factor=3.0) -> np.ndarray:
    """
    Wrapper that calls the numba version, then converts uint8 => bool.
    """
    outlier_uint8 = local_outlier_detection_nb(image, window_size, sigma_factor)
    return (outlier_uint8 == 1)


# 3. Edge Outlier Detection
def edge_outlier_detection(image: np.ndarray,
                           window_size=3,
                           sigma_factor=3.0) -> np.ndarray:
    """
    Example custom 'edge' outlier detection: 
    Flag voxels within 3 voxels of any boundary that exceed some threshold.
    """
    zdim, ydim, xdim = image.shape
    outliers = np.zeros((zdim, ydim, xdim), dtype=bool)
    
    # Mark boundary region
    margin = 3
    outliers[:margin, :, :] = True
    outliers[-margin:, :, :] = True
    outliers[:, :margin, :] = True
    outliers[:, -margin:, :] = True
    outliers[:, :, :margin] = True
    outliers[:, :, -margin:] = True
    
    # Now, define some intensity threshold for the boundary. For example:
    threshold = 0.4
    # Combine the boundary condition with the intensity check
    return outliers & (image > threshold)


# 4. Combined Outlier Detection
def combined_outlier_detection(image: np.ndarray,
                               global_factor: float = 3.0,
                               global_percentile: float = 90.0,
                               local_window: int = 3,
                               local_sigma: float = 3.0) -> np.ndarray:
    """
    Combine global and local outlier detection. Returns a boolean mask
    flagged if EITHER method detects an outlier.
    """
    global_mask = global_outlier_detection(image, factor=global_factor,
                                           percentile=global_percentile)
    local_mask = local_outlier_detection(image, window_size=local_window,
                                         sigma_factor=local_sigma)
    return global_mask | local_mask


# 5. Analyze Outlier Masks
def analyze_outlier_masks(image: np.ndarray,
                          global_mask: np.ndarray,
                          local_mask: np.ndarray,
                          edge_mask: np.ndarray) -> np.ndarray:
    """
    Print stats about how many voxels are flagged by each method and by both.
    Then return a combined mask for convenience.
    """
    # Example region focusing (like top/bottom slices or edges)
    outliers_region = np.zeros_like(image, dtype=bool)
    # Let’s define a 5-voxel margin:
    margin = 5
    outliers_region[:margin, :, :] = True
    outliers_region[-margin:, :, :] = True
    outliers_region[:, :margin, :] = True
    outliers_region[:, -margin:, :] = True
    outliers_region[:, :, :margin] = True
    outliers_region[:, :, -margin:] = True
    
    # Summaries
    n_global = np.sum(global_mask)
    n_edge = np.sum(edge_mask)
    # Local only in the region
    n_local = np.sum(local_mask & outliers_region)
    
    # Union of all
    combined_mask = (global_mask | edge_mask | (local_mask & outliers_region))
    n_combined = np.sum(combined_mask)
    
    print("Outlier Stats:")
    print(f"  Global outliers: {n_global}")
    print(f"  Edge outliers:   {n_edge}")
    print(f"  Local outliers:  {n_local}")
    print(f"  Combined union:  {n_combined}")
    
    return combined_mask


# 6. Iterative Removal of Outliers
def remove_outliers_iteratively(image: np.ndarray,
                                max_iters: int = 5,
                                global_factor: float = 3.0,
                                global_percentile: float = 95.0,
                                local_window: int = 3,
                                local_sigma: float = 3.0,
                                edge_thresh: float = 0.4) -> np.ndarray:
    """
    Repeatedly detect outliers (global + local + edge) and remove them 
    until no outliers remain or we exceed max_iters.
    
    'Remove' can be done via setting them to the local mean or zero, etc.
    For simplicity, we'll just set them to local mean in 3x3x3. 
    """
    patched = image.copy()
    
    for iteration in range(max_iters):
        global_mask = global_outlier_detection(patched, factor=global_factor,
                                               percentile=global_percentile)
        local_mask = local_outlier_detection(patched, window_size=local_window,
                                             sigma_factor=local_sigma)
        
        # For edge outliers, let's do a simpler approach:
        zdim, ydim, xdim = patched.shape
        edge_mask = np.zeros((zdim, ydim, xdim), dtype=bool)
        margin = 3
        edge_mask[:margin, :, :] = True
        edge_mask[-margin:, :, :] = True
        edge_mask[:, :margin, :] = True
        edge_mask[:, -margin:, :] = True
        edge_mask[:, :, :margin] = True
        edge_mask[:, :, -margin:] = True
        
        # Combine
        combined_mask = global_mask | (local_mask) | (edge_mask & (patched > edge_thresh))
        
        num_outliers = np.sum(combined_mask)
        print(f"Iteration {iteration+1}: found {num_outliers} outliers.")
        print(patched[0, 71, 44])
        if num_outliers == 0:
            print("No more outliers. Stopping early.")
            break
        
        # Inpaint the outliers with a local-mean approach (3x3x3).
        patched = inpaint_with_local_mean(patched, combined_mask)
        patched = (patched - patched.min()) / (patched.max() - patched.min() + 2e-8)
        # check nan and inf and raise error
        if np.isnan(patched).any() or np.isinf(patched).any():
            raise ValueError("Inpainting resulted in NaN or Inf values.")
    return patched

@numba.njit
def inpaint_with_local_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Inpaint outliers in 'mask' by replacing them with the average of 
    their non-outlier neighbors in a 3D 3x3x3 region.
    
    This is a simpler in-place method using numba. 
    We do a two-pass approach to avoid partial updates:
      1. Collect all new values in arrays.
      2. Write them back.
    """
    zdim, ydim, xdim = image.shape
    # We'll store replacements in parallel arrays:
    positions = []
    replacements = []
    
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if mask[z, y, x]:
                    # gather neighbors
                    sum_val = 0.0
                    count = 0
                    for zz in range(z-1, z+2):
                        for yy in range(y-1, y+2):
                            for xx in range(x-1, x+2):
                                if 0 <= zz < zdim and 0 <= yy < ydim and 0 <= xx < xdim:
                                    if not mask[zz, yy, xx]:
                                        sum_val += image[zz, yy, xx]
                                        count += 1
                    if count > 0:
                        new_val = sum_val / count
                    else:
                        new_val = 0.0
                    positions.append((z, y, x))
                    replacements.append(new_val)
    
    # second pass: write them back
    for i in range(len(positions)):
        z, y, x = positions[i]
        image[z, y, x] = replacements[i]
    
    return image





if __name__ == "__main__":
    import os
    # Example usage:
    # image = np.random.rand(100, 100, 100)  # Replace with your actual image
    # outliers = combined_outlier_detection(image)
    # print("Detected outliers:", np.sum(outliers))
    
    input_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed"
    output_dir = r"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm"
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Load the array
            image = np.load(input_path)
            
            # Remove outliers
            patched_image = remove_outliers_iteratively(image)
            
            # Save the patched image
            np.save(output_path, patched_image)
            print(f"Processed {filename} and saved to {output_path}")