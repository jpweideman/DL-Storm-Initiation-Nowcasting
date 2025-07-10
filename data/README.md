# Data Folder Structure

This folder contains all data used in the pipeline.

## Folders

- **raw/**
  - Place your original, unmodified radar data files here (e.g., `.h5` files).
  - These files are never overwritten or modified by scripts.
  - Example: `data/raw/2024/08/14/scan-sidpol-120km-*.h5`

- **intermediate/**
  - Contains intermediate processed data chunks generated from raw data.
  - Each subdirectory mirrors the structure of `raw/` and contains:
    - `data.npy` - Processed radar data chunks
    - `filenames.json` - Corresponding original filenames
  - This folder is used for memory-efficient processing of large datasets.
  - Example: `data/intermediate/2024/08/14/data.npy`

- **processed/**
  - Contains final training-ready data files.
  - These are the files used by all training and evaluation scripts.
  - Clean, single large files for training input.
  - Example: `data/processed/{ZH_radar_dataset.npy, ZH_radar_filenames.json}`

## Data Flow

1. **Raw Data**: Place raw `.h5` files in `data/raw/` (organized by year/month)
2. **Intermediate Processing**: Run `src/data/data_processing.py` to create chunks in `data/intermediate/`
3. **Final Joining**: Run `src/data/join_processed_data.py` to create final files in `data/processed/`
4. **Training**: Use files from `data/processed/` for model training and evaluation


See [src/data/README.md](../src/data/README.md) for detailed processing script documentation. 