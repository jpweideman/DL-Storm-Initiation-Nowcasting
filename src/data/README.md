# Data Processing Scripts

This directory contains scripts for processing raw radar data from `.h5` files into training-ready formats.

## Overview

The data processing workflow consists of two steps:

1. **`data_processing.py`** - Processes raw `.h5` files into intermediate chunks
2. **`join_processed_data.py`** - Joins all intermediate chunks into final training files

## Workflow Summary

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| 1 | `data/raw/*.h5` | `data/intermediate/*/data.npy` | Process raw files into memory-efficient chunks |
| 2 | `data/intermediate/*/data.npy` | `data/processed/ZH_radar_dataset.npy` | Join all chunks into single training file |

## Scripts

### 1. data_processing.py

Processes raw `.h5` radar files into intermediate `.npy` chunks organized by directory structure.

**Input:** Raw `.h5` files in `data/raw/` (organized by year/month subdirectories)
**Output:** Intermediate chunks in `data/intermediate/` (preserving directory structure)

**Features:**
- Processes each `.h5` file into 14 radar scans (360×240 pixels each)
- Cleans data by replacing invalid values (96.00197) with 0
- Pads or crops images to target size (360×240)
- Saves chunks as `.npy` files with corresponding filename lists as `.json`
- Memory-efficient processing for large datasets

**Usage:**
```bash
python src/data/data_processing.py
```

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

**Features:**
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
