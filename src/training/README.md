# Training & Evaluation Scripts

This folder contains scripts for training and evaluating different deep learning models for radar nowcasting.

## Available Training Scripts

- **train_conv_lstm.py** — Train a ConvLSTM model.
- **train_conv_lstm_nonnormalized.py** — Train ConvLSTM on non-normalized data.
- **train_trajGRU.py** — Train a Trajectory GRU model.
- **train_trajGRU_baseline.py** — Train a symmetric TrajGRU encoder-decoder model (no U-Net, no skip connections).
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
- Add `--no_wandb` to disable Weights & Biases logging.

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

## Patch-based Training: The `--use_patches` Argument

All training and testing scripts support patch-based training and evaluation via the `--use_patches` argument. This enables the model to focus on spatial sub-regions (patches) of the radar data that are most relevant for learning, rather than always using the full spatial field.

**How to use:**

Add `--use_patches True` to your training or testing command, and optionally adjust the patch extraction parameters:

- `--patch_size`: Size of each spatial patch (default: 64)
- `--patch_stride`: Stride for patch extraction (default: 32)
- `--patch_thresh`: Minimum normalized reflectivity for a pixel to be considered active (default: 0.35)
- `--patch_frac`: Minimum fraction of active pixels in a patch (default: 0.05)

**Example:**

```bash
python src/training/train_unet_3D_cnn.py train \
  ... (other arguments) ... \
  --use_patches True \
  --patch_size 64 \
  --patch_stride 32 \
  --patch_thresh 0.35 \
  --patch_frac 0.05
```

**Why use patch-based training?**

- **Efficiency:** Training on patches allows the model to see more diverse and localized weather events per epoch, improving data efficiency and convergence speed.
- **Focus on storms:** By extracting only patches with a significant fraction of high-reflectivity pixels, the model focuses on learning from regions with active weather (e.g., storms), rather than background or empty areas.
- **Memory savings:** Patch-based training reduces GPU memory requirements, enabling larger batch sizes or higher-resolution inputs.
- **Better generalization:** The model learns to predict local storm structures and dynamics, which can improve generalization to new events and locations.

Patch-based training is highly recommended for radar nowcasting tasks, especially when storms are sparse in space and time.

## Outputs
- **Checkpoints**: Saved in the run directory.
- **Arguments**: Saved as `args.json` in the run directory for reproducibility.
- **Results**: Results saved in `results/` inside the run directory.
- **Predictions**: Large data arrays from testing can be saved in a separate directory using `--predictions_dir`.

See the main [README.md](../../README.md) for the full pipeline. 