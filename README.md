# Radar Precipitation Nowcasting Pipeline

This repository provides a modular, reproducible pipeline for radar-based precipitation nowcasting using deep learning models. 

## Folder Structure

- `data/raw/` — Place your original raw radar data files here (`.h5` files). See [data/README.md](data/README.md). 
- `data/processed/` — Processed data files (e.g., `.npy`, `.json`) generated from raw data. Used as input for training. 
- `src/data/` — Data processing scripts. See [src/data/README.md](src/data/README.md).
- `src/models/` — Model architecture definitions. See [src/models/README.md](src/models/README.md).
- `src/training/` — Training scripts for different models. See [src/training/README.md](src/training/README.md).
- `experiments/runs/` — Each training run saves checkpoints, args, and results here.
- `experiments/wandb/` — [Weights & Biases](https://wandb.ai/) experiment logs (if enabled during training).

## Quickstart: End-to-End Example

1. **Prepare Data**
   - Place your raw `.h5` radar files in `data/raw/` (an example dataset is already provided).

2. **Process Data**
   - Run a data processing script to convert raw data to `.npy` in `data/processed/`:
     ```bash
     python src/data/data_processing.py --input_dir data/raw/ --output_dir data/processed/
     # or see src/data/README.md for other scripts and options
     ```

3. **Train a Model**
   - Run a training script (e.g., UNet 3D CNN):
     ```bash
     python src/training/train_unet_3D_cnn.py --data_dir data/processed/ --run_dir experiments/runs/unet3d_example
     # Add other CLI arguments as needed (see src/training/README.md)
     ```
   - Arguments used for the run are saved as `args.json` in the run directory.

4. **Evaluate & Save Results**
   - Evaluation results  are saved in `results/` inside the run directory.
   - Large prediction/target files can be saved elsewhere using `--predictions_dir`.

5. **Track Experiments**
   - If using Weights & Biases, logs are saved in `experiments/wandb/`.

<!-- ## More Information
- See [src/data/README.md](src/data/README.md) for data processing details.
- See [src/training/README.md](src/training/README.md) for training and evaluation script usage.
- See [src/models/README.md](src/models/README.md) for available model architectures.
- See [data/README.md](data/README.md) for data folder explanations.
- See [experiments/README.md](experiments/README.md) for experiment output structure. -->







