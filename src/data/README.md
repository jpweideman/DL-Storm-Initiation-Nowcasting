# Data Processing Scripts

This directory contains scripts for processing raw radar data from `.h5` files into training-ready formats.

## Overview

The data processing workflow consists of two steps:

1. **`data_processing.py`** - Processes raw `.h5` files into intermediate chunks
2. **`join_processed_data.py`** - Joins all intermediate chunks into final training files

## Scripts

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

# Custom parameters for different data formats
python src/data/data_processing.py --target_height 360 --target_width 320 --num_channels 8 --variable ZV --noise_value 0.0
```

**Command-line Arguments:**
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

**Output:** Final training-ready files in `data/processed/`

- Concatenates all intermediate `.npy` files into a single large array
- Combines all filename lists into a single JSON file
- Uses memory mapping for handling of large datasets
- Creates clean final output in `data/processed/`

**Usage:**
```bash
python src/data/join_processed_data.py
```

**Output Files:**
- `data/processed/ZH_radar_dataset.npy` - Complete processed dataset
- `data/processed/ZH_radar_filenames.json` - Complete list of original filenames
