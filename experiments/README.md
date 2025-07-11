# Experiments Folder

This folder organizes all experiment outputs for training, evaluation, and tracking.

## Structure

- **runs/**
  - Each subfolder is a separate experiment run (e.g., `unet3d_example/`).
  - Contains:
    - Model checkpoints
    - `args.json` (CLI arguments for reproducibility)
    - `results/` (evaluation metrics etc.)
    - Other run-specific outputs

- **wandb/**
  - Contains [Weights & Biases](https://wandb.ai/) logs for experiment tracking (if enabled).

## Usage
- **Important:** To follow the intended structure of this repository, you must ensure that the `--save_dir` (for training) and `--run_dir` (for training/testing) arguments point to a subdirectory of `experiments/runs/`.
    - Example: `--save_dir experiments/runs/my_experiment` and `--run_dir experiments/runs/my_experiment`
- All results and checkpoints for that run will be saved in the specified folder.
- Use `--predictions_dir` to save large prediction/target files elsewhere if needed.

See the main [README.md](../README.md) for the full workflow. 