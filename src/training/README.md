# Training & Evaluation Scripts

This folder contains scripts for training and evaluating different deep learning models for radar nowcasting.

## Available Training Scripts

- **train_conv_lstm.py** — Train a ConvLSTM model.
- **train_trajGRU.py** — Train a Trajectory GRU model.
- **train_trajGRU_enc_dec.py** — Train a symmetric TrajGRU encoder-decoder model (no U-Net, no skip connections).
- **train_3D_cnn.py** — Train a 3D CNN model.
- **train_unet_3D_cnn.py** — Train a U-Net 3D CNN model.
- **train_unet_conv_lstm.py** — Train a U-Net ConvLSTM model.
- **train_unet_trajGRU.py** — Train a U-Net TrajGRU model.

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
  --train_val_test_split "(0.5,0.1,0.4)" \
  --early_stopping_patience 10 \
  --no_wandb
```

- All arguments used for the run are saved as `args.json` in the run directory for reproducibility.
- Validation metrics (CSI, HSS, B-MSE, MSE by dBZ bins) are automatically computed during training and saved to `results/best_validation_metrics.json` when a new best validation score is achieved.
- Use `--no_wandb` to disable Weights & Biases logging.
- To use [Weights & Biases](https://wandb.ai/) logging, add `--wandb_project "project-name"` to your command. This will log training metrics, model parameters, and enable experiment tracking.

## Example: Test a UNet 3D CNN Model

```bash
python src/training/train_unet_3D_cnn.py test \
  --npy_path data/processed/ZH_radar_dataset.npy \
  --run_dir experiments/runs/unet3dcnn_example \
  --seq_len_in 10 \
  --seq_len_out 1 \
  --train_val_test_split "(0.5,0.1,0.4)" \
  --batch_size 4 \
  --base_ch 64 \
  --bottleneck_dims "(32,)" \
  --kernel_size 3 \
  --which best \
  --device cpu \
  --save_arrays True \
  --predictions_dir predictions/unet3dcnn_example
```

- Test metrics (CSI, HSS, B-MSE, MSE by dBZ bins, confusion matrices) are automatically computed and saved to `results/test_metrics.json` in the run directory.
- For large datasets, it is possible to use the `--predictions_dir` argument to save prediction and target arrays in a separate directory (outside the run directory). 
This is useful if one wants to save the large output files to an external storage location.
- **Note**: The testing function returns arrays of **Composite Reflectivity (maximum reflectivity projection over altitudes)**. If your model outputs have multiple channels (e.g., different altitude levels), the testing function automatically reduces them to composite reflectivity.

## Patch-based Training (`--use_patches` argument):

All training and testing scripts support patch-based training via the `--use_patches` argument. This enables the model to focus on spatial sub-regions (patches) of the radar data that are most relevant for learning, rather than always using the full spatial field.

**How to use:**

Add `--use_patches True` to your training command, and optionally adjust the patch extraction parameters:

- `--patch_size`: Size of each spatial patch (default: 64)
- `--patch_stride`: Stride for patch extraction (default: 32)
- `--patch_thresh`: Minimum normalized reflectivity for a pixel to be considered active (dBZ, default: 35)
- `--patch_frac`: Minimum fraction of active pixels in a patch (default: 0.01)

**Example:**

```bash
python src/training/train_unet_3D_cnn.py train \
  ... (other arguments) ... \
  --use_patches True \
  --patch_size 64 \
  --patch_stride 32 \
  --patch_thresh 35 \
  --patch_frac 0.01
```

Patch-based training is highly recommended for radar nowcasting of storm initiation.

- **Focus on high reflectivity areas:** By extracting only patches with a significant fraction of high-reflectivity pixels, the model focuses on learning from regions with active weather, rather than background or empty areas.
- **Memory savings:** Patch-based training reduces memory requirements, since input dimensions are significantly smaller.
- **Training Speedup** Training is much faster, and often leads to better storm initiation forecasts. 

## Outputs
- **Checkpoints**: Saved in the run directory.
- **Arguments**: Saved as `{train/test}_args.json` in the run directory.
- **Results**: Results saved in `results/` inside the run directory.
- **Validation Metrics**: Automatically saved to `results/best_validation_metrics.json` when new best validation scores are achieved.
- **Test Metrics**: Automatically saved to `results/test_metrics.json` when running the test command. Contains comprehensive evaluation metrics including CSI, HSS, B-MSE, MSE by dBZ bins, and confusion matrices for all thresholds.
- **Predictions**: Large data arrays from testing can be saved in a separate directory using `--predictions_dir`.

See the main [README.md](../../README.md) for the full pipeline. 