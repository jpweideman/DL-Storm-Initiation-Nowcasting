# Experiments Folder

This folder organizes all experiment outputs for training, evaluation, and tracking.

## Structure

- **runs/**
  - Each subfolder is a separate experiment run (e.g., `unet3d_example/`).
  - Contains:
    - Model checkpoints
    - `args.json` (CLI arguments for reproducibility)
    - `results/` (evaluation metrics, plots, etc.)
    - Any other run-specific outputs

- **wandb/**
  - Contains [Weights & Biases](https://wandb.ai/) logs for experiment tracking (if enabled).

## Usage
- Specify `--run_dir` when running a training script to create a new run directory.
- All results and checkpoints for that run will be saved in the specified folder.
- Use `--predictions_dir` to save large prediction/target files elsewhere if needed.

See the main [README.md](../README.md) for the full workflow. 