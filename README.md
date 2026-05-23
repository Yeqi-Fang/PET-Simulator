# PET Reconstruction Pipeline

Graduation thesis project for simulating and reconstructing PET images with incomplete detector ring geometries. The pipeline generates synthetic PET data, processes sinograms, and reconstructs 3D volumes via OSEM.

## Project Structure

```
graduation-thesis2/
├── pet_simulator/          # PET scanner simulation package
├── pipeline/               # Data generation and preprocessing
├── reconstruction/         # OSEM reconstruction scripts
├── analysis/               # Metrics, visualization, and notebooks
├── utils/                  # Helper and verification tools
├── scripts/                # Shell runner scripts (run from project root)
├── spare/                  # Archived legacy code (reference only)
├── log/                    # Timestamped output logs
├── paper_image/            # Publication-quality figures
├── detector_lut.txt        # Scanner geometry lookup table (364 crystals × 42 rings)
└── command.txt             # Command history and pipeline notes
```

---

## pet_simulator/

PET scanner simulation package (Python package with `__init__.py`).

| File | Description |
|------|-------------|
| `geometry.py` | Scanner geometry dataclass (radius 253.71 mm, 42 rings) |
| `simulator.py` | Monte Carlo event simulation engine, multiprocessing-capable |
| `numba_utils.py` | Numba-accelerated numerical kernels |
| `utils.py` | Save/load utilities for events and detector LUT |

---

## pipeline/

Data generation and preprocessing. Run scripts from the **project root**:

```bash
python pipeline/<script>.py [args]
```

| File | Description |
|------|-------------|
| `listmode_to_incomplete_new.py` | Filter complete listmode data by missing detector sectors to simulate incomplete rings |
| `sinogram_reconstruction.py` | Reconstruct 3D volumes directly from sinogram directories using OSEM |
| `smooth.py` | Apply Gaussian smoothing to sinogram `.npy` files |
| `split_sinogram_files.py` | Split full sinograms (1764 slices) into individual axial slice files |
| `renamee_new.py` | Organize complete/incomplete sinograms into `train/test` splits with standardized naming |
| `prepare_dataset.py` | Prepare datasets for model training |
| `resize_normalize.py` | Resize and normalize PET image volumes |
| `rotate_new_data.py` | Rotate image volumes to canonical orientation |

### Typical pipeline order

```
listmode_to_incomplete_new.py
    → sinogram_reconstruction.py
        → smooth.py
            → split_sinogram_files.py
                → renamee_new.py
```

---

## reconstruction/

OSEM reconstruction from assembled sinogram slices.

| File | Description |
|------|-------------|
| `div_osem_masked.py` | **Main script.** Assemble divided sinogram slices, apply missing-sector mask, then OSEM reconstruct. Supports AD/MCI/Healthy/predicted data formats. |
| `div_to_osem.py` | Same as above without masking; used for complete-ring reconstruction. |
| `generate_reconstruct.py` | End-to-end simulation + reconstruction pipeline (uses `pet_simulator`). |
| `reconstruction_all.py` | Batch reconstruction directly from listmode `.npz` files. |

---

## analysis/

Metrics computation, quality visualization, and paper figures.

| File | Description |
|------|-------------|
| `calculate_metric.ipynb` | Compute SSIM, PSNR, MSE across all reconstruction results |
| `paper_plot.ipynb` | Generate publication-quality figures for the thesis |
| `view_sinogram.ipynb` | Interactive sinogram viewer |
| `enhanced_visualization.py` | Multi-view comparison of complete vs. incomplete ring reconstructions |
| `compare_reconstruction_restoration.py` | Side-by-side: original reconstruction vs. model-restored image |
| `compute_missing_coincidence.py` | Compute coincidence-line statistics for missing detector sectors |
| `outlier_detection.py` | Global, local, and edge outlier detection and removal |

---

## utils/

Verification and conversion tools.

| File | Description |
|------|-------------|
| `verify_mask.py` | Visualize and verify mask application on sinogram pairs |
| `verify_pairs.py` | Check all complete/incomplete `.npy` pairs for consistency |
| `find_duplicates.py` | Detect duplicate data files across directories |
| `npy2mat.py` | Convert `.npy` arrays to MATLAB `.mat` format |
| `npz2npy.py` | Extract `.npy` files from compressed `.npz` archives |

---

## scripts/

Shell scripts for batch execution on Linux/remote servers. Always run from the **project root**:

```bash
bash scripts/run_osem.sh
```

| File | Description |
|------|-------------|
| `run_osem.sh` | Batch OSEM reconstruction for AD, MCI, and Healthy datasets (complete + predicted) |
| `wait_and_div_to_osem.sh` | Wait for a running process (by PID), then launch `div_to_osem.py` |
| `run_incomplete_conversion.sh` | Run the incomplete ring data generation pipeline |

---

## spare/

Archived legacy and experimental code kept for reference. Not part of the active pipeline.

Contents include early simulation notebooks, differential-privacy PET experiments (`DP-PET/`), MATLAB reconstruction scripts (`matlab/`), and previous iterations of core scripts (`old/`).

---

## Data flow

```
3D activity images (.npy)
    │
    ▼ pet_simulator / generate_reconstruct.py
Complete listmode events (.npz)
    │
    ├──▶ pipeline/sinogram_reconstruction.py ──▶ Complete sinograms
    │
    └──▶ pipeline/listmode_to_incomplete_new.py ──▶ Incomplete listmode
                                                         │
                                                         ▼
                                                 pipeline/sinogram_reconstruction.py
                                                         │
                                                         ▼
                                                 Incomplete sinograms
    │
    ▼ pipeline/smooth.py ──▶ pipeline/split_sinogram_files.py
Smoothed divided sinograms (per axial slice)
    │
    ▼ pipeline/renamee_new.py
train/complete_i.npy + train/incomplete_i.npy
    │
    ▼ reconstruction/div_osem_masked.py (or div_to_osem.py)
Reconstructed 3D volumes (80×128×128)
    │
    ▼ analysis/calculate_metric.ipynb
SSIM / PSNR / MSE metrics
```

---

## Dependencies

- Python 3.9+
- `pytomography` (OSEM reconstruction)
- `torch`, `numpy`, `scipy`, `matplotlib`
- `numba` (simulation acceleration)
- `tqdm`

Install:
```bash
pip install torch numpy scipy matplotlib numba tqdm
# install pytomography per its documentation
```
