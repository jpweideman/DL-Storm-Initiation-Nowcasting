# Data Processing Scripts

This folder contains scripts for converting raw radar data into processed formats suitable for model training and evaluation.

## Scripts

- **data_processing.py**
  - Main script for processing raw `.h5` radar files into `.npy` and `.json` files.
  - **Usage:**
    ```bash
    python data_processing.py --input_dir ../../data/raw/ --output_dir ../../data/processed/
    ```
  - **Arguments:**
    - `--input_dir`: Path to raw `.h5` files.
    - `--output_dir`: Path to save processed `.npy`/`.json` files.

- **data_processing_complete_data.py**
  - Processes the complete dataset, possibly with additional features or for full dataset runs.
  - **Usage:**
    ```bash
    python data_processing_complete_data.py --input_dir ../../data/raw/ --output_dir ../../data/processed/
    ```
  - **Arguments:** Same as above.

- **join_processed_data.py**
  - Joins or merges processed data files (e.g., for combining multiple `.npy`/`.json` files).
  - **Usage:**
    ```bash
    python join_processed_data.py --input_dir ../../data/processed/ --output_file ../../data/processed/combined.npy
    ```
  - **Arguments:**
    - `--input_dir`: Directory with processed files to join.
    - `--output_file`: Output file for the joined data.

## Output
- Processed `.npy` and `.json` files are saved in `data/processed/`.
- These files are used as input for all training scripts.

See the main [README.md](../../README.md) for the full pipeline. 