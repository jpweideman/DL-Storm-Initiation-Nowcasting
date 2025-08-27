# Data Folder Structure

This folder contains all data used in the pipeline.

## Folders

- **raw/**
  - Place original, unmodified radar data files here (e.g., `.h5` files).
  - Organized by `year/month/day/` subdirectories.

- **intermediate/**
  - Contains intermediate processed data chunks generated from raw data.
  - Each subdirectory mirrors the structure of `raw/` and contains:
    - `data.npy` - Processed radar data chunks
    - `filenames.json` - Corresponding original filenames
  - This folder is used as an intermediate step for memory-efficient processing of large datasets.

- **processed/**
  - Contains final training-ready data files.
  - These are the files used by all training and evaluation scripts.
  - Clean, single large files for training input.

## File Usage

1. **Raw Data**: Place raw `.h5` files in `data/raw/` (organized by year/month/day).
2. **Intermediate Processing**: Run `src/data/data_processing.py` to create chunks in `data/intermediate/`.
3. **Joining**: Run `src/data/join_processed_data.py` to create raw dataset `data/processed/ZH_radar_dataset_raw.npy`.
4. **Ground Clutter Removal**: Run `src/data/remove_ground_clutter.py` to create final cleaned dataset `data/processed/ZH_radar_dataset.npy`.
5. **Training**: Use `data/processed/ZH_radar_dataset.npy` for model training and evaluation.

See [src/data/README.md](../src/data/README.md) for detailed data processing steps. 