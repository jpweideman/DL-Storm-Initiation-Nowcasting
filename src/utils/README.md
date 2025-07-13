# Storm Utils

This folder contains utility functions for storm detection, animation, and evaluation in radar nowcasting experiments.

## Structure and Organization

- **storm_utils.py**: Core storm detection and evaluation functions (e.g., `detect_storms`, `detect_new_storm_formations`, `evaluate_new_storm_predictions`, `count_storms_by_section`).
- **storm_animation_utils.py**: All animation and visualization functions (e.g., `animate_storms`, `animate_new_storms`, `animate_storms_polar`, `animate_storms_polar_comparison`).
- **storm_section_counter.py**: Script for counting storms and new storms in temporal sections of data.

## Main Features

- **Storm Detection**: Identify storms and new storm initiations in radar reflectivity data.
- **Animation**: Visualize storms and new storm formations over time.
- **Evaluation**: Quantitatively compare predicted and true storm initiations.
- **Section Analysis**: Count storms and new storms in different sections of the data.

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
- `--targets`: Path true reflectivity `.npy` file
- `--out`: Output JSON file for evaluation results
- `--reflectivity_threshold`: dBZ threshold for storm detection (default: 45)
- `--area_threshold`: Minimum storm area in pixels (default: 15)
- `--dilation_iterations`: Dilation iterations for storm region merging (default: 5)
- `--overlap_threshold`: Overlap ratio for matching storms (default: 0.2)

## CLI Usage: Count Storms by Data Sections

Analyze storm and new storm counts across temporal sections of radar data:

```bash
python src/utils/storm_section_counter.py \
  --npy_path data/processed/ZH_radar_dataset.npy \
  --interval_percent 5 \
  --batch_size 10 \
  --reflectivity_threshold 45 \
  --area_threshold 15 \
  --dilation_iterations 5 \
  --overlap_threshold 0.1 \
  --out results/storm_section_counts.json
```

- `--npy_path`: Path to radar data `.npy` file (shape: T, H, W or N, C, H, W)
- `--interval_percent`: Section size as percentage of total data length (default: 5)
- `--batch_size`: Number of frames to process at once (for memory efficiency) (default: 10)
- `--reflectivity_threshold`: dBZ threshold for storm detection (default: 45)
- `--area_threshold`: Minimum storm area in pixels (default: 15)
- `--dilation_iterations`: Dilation iterations for storm region merging (default: 5)
- `--overlap_threshold`: Overlap threshold for new storm detection (default: 0.1)
- `--out`: Optional output JSON file for results

## Main Python Functions

### In `storm_utils.py` (detection & evaluation)
- `detect_storms(data, reflectivity_threshold, area_threshold, dilation_iterations)`
  - Detects storms in each frame of radar data.
- `detect_new_storm_formations(data, reflectivity_threshold, area_threshold, dilation_iterations, overlap_threshold)`
  - Identifies new storm initiations over time.
- `evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold)`
  - Compares predicted and true new storm initiations - returns metrics.
- `count_storms_by_section(data, interval_percent, batch_size, ...)`
  - Count storms and new storms in temporal sections of data.

### In `storm_animation_utils.py` (visualization)
- `animate_storms(data, ...)`
  - Create matplotlib animation of storms over time.
- `animate_new_storms(data, new_storms_result)`
  - Animate new storm initiations over time.
- `animate_storms_polar(data, ...)` and `animate_storms_polar_comparison(true_data, pred_data, ...)`
  - Polar coordinate visualizations for radar data.

## Example Notebook Code

```python
from src.utils import storm_utils
from src.utils import storm_animation_utils

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


# Animate storms (in a notebook)
ani = storm_animation_utils.animate_storms(preds)
from IPython.display import HTML
display(HTML(ani.to_jshtml()))
```
