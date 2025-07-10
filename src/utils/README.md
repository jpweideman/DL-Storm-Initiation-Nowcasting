# Storm Utils

This folder contains utility functions for storm detection, animation, and evaluation in radar nowcasting experiments.

## Main Features

- **Storm Detection**: Identify storms and new storm initiations in radar reflectivity data.
- **Animation**: Visualize storms and new storm formations over time.
- **Evaluation**: Quantitatively compare predicted and true storm initiations.
- **CLI**: Command-line interface for evaluating predictions and saving results as JSON.

## CLI Usage: Evaluate Storm Initiation Predictions

After running testing and saving predictions/targets as `.npy` files, evaluate storm initiations with:

```bash
python src/utils/storm_utils.py \
  --preds predictions/unet3dcnn_example/test_preds_dBZ.npy \
  --targets predictions/unet3dcnn_example/test_targets_dBZ.npy \
  --out experiments/runs/unet3dcnn_example/results/storm_eval.json \
  --reflectivity_threshold 45 \
  --area_threshold 15 \
  --dilation_iterations 5 \
  --overlap_threshold 0.2
```

- `--preds`: Path to predicted reflectivity `.npy` file (shape: N, C, H, W or N, H, W)
- `--targets`: Path to ground truth reflectivity `.npy` file
- `--out`: Output JSON file for evaluation results
- `--reflectivity_threshold`: dBZ threshold for storm detection (default: 45)
- `--area_threshold`: Minimum storm area in pixels (default: 15)
- `--dilation_iterations`: Dilation iterations for storm region merging (default: 5)
- `--overlap_threshold`: Overlap ratio for matching storms (default: 0.2)

## Main Python Functions

- `detect_storms(data, reflectivity_threshold, area_threshold, dilation_iterations)`
  - Detects storms in each frame of radar data.
- `detect_new_storm_formations(data, reflectivity_threshold, area_threshold, dilation_iterations, overlap_threshold)`
  - Identifies new storm initiations over time.
- `evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold)`
  - Compares predicted and true new storm initiations, returns metrics.
- `animate_storms(data, ...)` and `animate_new_storms(data, new_storms_result)`
  - Create matplotlib animations of storms and new storm initiations.

## Example (Python API)

```python
from src.utils import storm_utils

# Load predictions and targets (N, C, H, W or N, H, W)
preds = np.load('experiments/runs/unet3dcnn_example/test_preds_dBZ.npy')
targets = np.load('experiments/runs/unet3dcnn_example/test_targets_dBZ.npy')

# Reduce to (T, H, W) if needed
if preds.ndim == 4:
    preds = np.max(preds, axis=1)
    targets = np.max(targets, axis=1)

# Detect new storms
pred_new_storms = storm_utils.detect_new_storm_formations(preds)
target_new_storms = storm_utils.detect_new_storm_formations(targets)

# Evaluate
metrics = storm_utils.evaluate_new_storm_predictions(pred_new_storms, target_new_storms)
print(metrics)
```

See the function docstrings in `storm_utils.py` for more details and advanced options. 