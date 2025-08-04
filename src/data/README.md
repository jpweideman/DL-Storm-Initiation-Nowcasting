# Data Processing Scripts

This folder contains scripts for processing raw radar data from `.h5` files into training-ready formats.

## Overview

The data processing workflow consists of three steps:

1. **`data_processing.py`** - Processes raw `.h5` files into intermediate chunks
2. **`join_processed_data.py`** - Joins all intermediate chunks into final training files
3. **`remove_ground_clutter.py`** - Removes ground clutter from the final dataset

### 1. data_processing.py

Processes raw `.h5` radar files into intermediate `.npy` chunks organized by directory structure.

**Input:** Raw `.h5` files in `data/raw/` (organized by year/month subdirectories)

**Output:** Intermediate chunks in `data/intermediate/` (preserving directory structure)

- Processes each `.h5` file into radar scans (configurable number of channels)
- Cleans data by replacing invalid values with 0 (configurable noise value)
- Pads or crops images to target size (configurable dimensions)
- Saves chunks as `.npy` files with corresponding filename lists as `.json`
- Memory-efficient processing for large datasets

**Usage:**
```bash
# Default usage (for the provided example dataset)
python src/data/data_processing.py
```

**Command-line Arguments:**
- `--input_dir`: Input directory containing HDF5 files (default: data/raw)
- `--output_dir`: Output directory for intermediate processed data (default: data/intermediate)
- `--target_height`: Target height for output arrays (default: 360)
- `--target_width`: Target width for output arrays (default: 240)
- `--num_channels`: Number of channels/scans to process (default: 14)
- `--variable`: Variable to extract from scans (default: "ZH")
- `--noise_value`: Noise value to replace with 0 (default: 96.00197)

**Output Structure:**
```
data/intermediate/
├── 2020/
│   ├── 01/
│   │   ├── data.npy          # Processed radar data chunks
│   │   └── filenames.json    # Corresponding original filenames
│   └── 02/
│       ├── data.npy
│       └── filenames.json
└── 2021/
    └── ...
```

### 2. join_processed_data.py

Joins all intermediate chunks into a single large training dataset.

**Input:** Intermediate chunks in `data/intermediate/`

**Output:** Raw processed files in `data/processed/`

- Concatenates all intermediate `.npy` files into a single large array
- Combines all filename lists into a single JSON file
- Uses memory mapping for handling of large datasets
- Creates output in `data/processed/`

**Usage:**
```bash
python src/data/join_processed_data.py
```

**Command-line Arguments:**
- `--input_dir`: Input directory containing processed data (default: data/intermediate)
- `--output_dir`: Output directory for final dataset (default: data/processed)
- `--output_name`: Output filename for the dataset (default: ZH_radar_dataset_raw.npy)

**Output Files:**
- `data/processed/ZH_radar_dataset_raw.npy` - Complete processed dataset (before ground clutter removal)
- `data/processed/ZH_radar_filenames.json` - Complete list of original filenames

### 3. remove_ground_clutter.py

Removes ground clutter from the radar dataset using height-based masking.

**Input:** Processed dataset from `data/processed/ZH_radar_dataset_raw.npy`

**Output:** Cleaned dataset in `data/processed/ZH_radar_dataset.npy`

- Calculates height above ground level (AGL) for each radar pixel using the 4/3 Earth radius model
- Creates a mask for data below the specified clutter height threshold
- Sets masked data to 0

**Usage:**
```bash
# Default usage (for the provided example dataset)
python src/data/remove_ground_clutter.py
```

**Command-line Arguments:**
- `--input_file`: Path to input radar data file (default: data/processed/ZH_radar_dataset_raw.npy)
- `--output_file`: Path to output cleaned radar data file (default: data/processed/ZH_radar_dataset.npy)
- `--clutter_height`: Height above ground level below which to set data to 0 (km, default: 1.0)
- `--radar_height_above_ground`: Height of radar antenna above ground (m, default: 38.0 for KITradar)
- `--elevations`: Comma-separated list of elevation angles in degrees (default: KITradar elevations)
- `--max_range`: Maximum range in kilometers (default: 120.0)
- `--range_resolution`: Range resolution in meters (default: 500.0)
- `--chunk_size`: Number of time steps to process at once (default: 100)
