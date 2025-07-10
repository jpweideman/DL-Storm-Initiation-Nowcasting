# Training & Evaluation Scripts

This folder contains scripts for training and evaluating different deep learning models for radar nowcasting.

## Available Training Scripts

- **train_conv_lstm.py** — Train a ConvLSTM model.
- **train_conv_lstm_nonnormalized.py** — Train ConvLSTM on non-normalized data.
- **train_trajGRU.py** — Train a Trajectory GRU model.
- **train_3D_cnn.py** — Train a 3D CNN model.
- **train_unet_3D_cnn.py** — Train a U-Net 3D CNN model.
- **train_unet_conv_lstm.py** — Train a U-Net ConvLSTM model.

## Usage Example
```bash
# With wandb logging (default)
python train_unet_3D_cnn.py --data_dir ../../data/processed/ --run_dir ../../experiments/runs/unet3d_example

# Without wandb logging
python train_unet_3D_cnn.py --data_dir ../../data/processed/ --run_dir ../../experiments/runs/unet3d_example --no_wandb
```

## Common Arguments
- `--data_dir`: Path to processed data (default: `data/processed/`).
- `--run_dir`: Directory to save checkpoints, args, and results.
- `--predictions_dir`: (Optional) Directory to save large prediction/target files.
- `--no_wandb`: Disable wandb logging (useful for local training without experiment tracking).
- Additional model/training hyperparameters (see `--help` for each script).

## Outputs
- **Checkpoints**: Saved in the run directory.
- **Arguments**: Saved as `args.json` in the run directory for reproducibility.
- **Results**: Metrics, plots, and evaluation outputs saved in `results/` inside the run directory.
- **Predictions**: Large files can be saved in a separate directory using `--predictions_dir`.

## Reproducibility
- All arguments are saved for each run.
- Use run directories to keep experiments organized.

See the main [README.md](../../README.md) for the full pipeline. 