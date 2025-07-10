# Training & Evaluation Scripts

This folder contains scripts for training and evaluating different deep learning models for radar nowcasting.

## Available Training Scripts

- **train_conv_lstm.py** — Train a ConvLSTM model.
- **train_conv_lstm_nonnormalized.py** — Train ConvLSTM on non-normalized data.
- **train_trajGRU.py** — Train a Trajectory GRU model.
- **train_3D_cnn.py** — Train a 3D CNN model.
- **train_unet_3D_cnn.py** — Train a U-Net 3D CNN model.
- **train_unet_conv_lstm.py** — Train a U-Net ConvLSTM model.

## Example: Train a UNet 3D CNN Model

```bash
python src/training/train_unet_3D_cnn.py train \
  --save_dir experiments/runs/unet3dcnn_example \
  --base_ch 64 \
  --bottleneck_dims "(32,)" \
  --kernel_size 3 \
  --npy_path data/processed/ZH_radar_dataset.npy \
  --seq_len_in 10 \
  --seq_len_out 1 \
  --batch_size 4 \
  --epochs 15 \
  --device cpu \
  --loss_name weighted_mse \
  --train_val_test_split "(0.6,0.2,0.2)" \
  --early_stopping_patience 10 \
  --no_wandb
```

- All arguments used for the run are saved as `args.json` in the run directory for reproducibility.
- Add `--no_wandb` to disable Weights & Biases logging (recommended for local or test runs).

## Example: Test a UNet 3D CNN Model

```bash
python src/training/train_unet_3D_cnn.py test \
  --npy_path data/processed/ZH_radar_dataset.npy \
  --run_dir experiments/runs/unet3dcnn_example \
  --seq_len_in 10 \
  --seq_len_out 1 \
  --train_val_test_split "(0.6,0.2,0.2)" \
  --batch_size 4 \
  --base_ch 64 \
  --bottleneck_dims "(32,)" \
  --kernel_size 3 \
  --which best \
  --device cpu \
  --save_arrays True \
  --predictions_dir predictions/unet3dcnn_example
```

- For large datasets, it is possible to use the `--predictions_dir` argument to save prediction and target arrays in a separate directory (outside the run directory). 
This is useful to avoid filling up the run directory with large files.
<!-- ## Common Arguments
- `--save_dir`: Directory to save checkpoints, args, and results.
- `--npy_path`: Path to processed radar data (default: `data/processed/ZH_radar_dataset.npy`).
- `--base_ch`: Base number of channels for U-Net encoder/decoder.
- `--bottleneck_dims`: Tuple/list of widths for 3D CNN bottleneck, e.g., "(32, 64, 32)".
- `--kernel_size`: Convolution kernel size (must be odd).
- `--seq_len_in`: Input sequence length (default: 10).
- `--seq_len_out`: Output sequence length (default: 1).
- `--batch_size`: Batch size (default: 4).
- `--epochs`: Number of training epochs (default: 15).
- `--device`: Device to train on ('cuda' or 'cpu').
- `--loss_name`: Loss function: mse or weighted_mse.
- `--train_val_test_split`: Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., "(0.7,0.15,0.15)".
- `--early_stopping_patience`: Number of epochs with no improvement before early stopping (default: 10).
- `--no_wandb`: Disable wandb logging. -->

## Outputs
- **Checkpoints**: Saved in the run directory.
- **Arguments**: Saved as `args.json` in the run directory for reproducibility.
- **Results**: Results saved in `results/` inside the run directory.
- **Predictions**: Large data arrays from testing can be saved in a separate directory using `--predictions_dir`.

See the main [README.md](../../README.md) for the full pipeline. 