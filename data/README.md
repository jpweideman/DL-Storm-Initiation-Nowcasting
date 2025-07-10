# Data Folder Structure

This folder contains all data used in the pipeline, organized for clarity and reproducibility.

## Folders

- **raw/**
  - Place your original, unmodified radar data files here (e.g., `.h5` files).
  - These files are never overwritten or modified by scripts.
  - Example: `data/raw/scan-sidpol-120km-*.h5`

- **processed/**
  - Contains all processed data files generated from the raw data (e.g., `.npy`, `.json`).
  - All training and evaluation scripts use files from this folder by default.
  - Example: `data/processed/ZH_radar_dataset_small.npy`, `ZH_radar_filenames.json`

## Data Flow
1. Place raw files in `data/raw/`.
2. Run a script from `src/data/` to generate processed files in `data/processed/`.
3. Use processed files for training and evaluation.

See [src/data/README.md](../src/data/README.md) for details on processing scripts. 