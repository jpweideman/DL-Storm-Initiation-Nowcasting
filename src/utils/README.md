# Storm Utils

This folder contains utility functions for storm detection, animation, and evaluation in radar nowcasting experiments.

## Structure and Organization

- **storm_utils.py**: Core storm detection and evaluation functions (e.g., `detect_storms`, `detect_new_storm_formations`, `evaluate_new_storm_predictions`, `count_storms_by_section`).
- **storm_animation_utils.py**: All animation and visualization functions (e.g., `animate_storms`, `animate_new_storms`, `animate_storms_polar`, `animate_storms_polar_comparison`).
- **storm_section_counter.py**: Script for counting storms and new storms in temporal sections of data.

## Main Features

- **Storm Detection**: Identify storms and new storm initiations in radar reflectivity data using **physical area calculations** that account for polar coordinate geometry.
- **Animation**: Visualize storms and new storm formations over time.
- **Evaluation**: Quantitatively compare predicted and true storm initiations.
- **Section Analysis**: Count storms and new storms in different sections of the data.
- **Forecasting Metrics**: Compute CSI, HSS, and B-MSE metrics for comprehensive model evaluation.

## Polar Coordinate Improvements

The storm detection now properly handles **radar polar coordinate geometry**:

- **Problem**: Traditional pixel-based storm detection doesn't account for the fact that radar data is in polar coordinates, where the same number of pixels represents different physical areas depending on distance from the radar.

- **Solution**: The `compute_polar_pixel_areas()` function calculates the physical area (in km²) of each pixel using polar geometry:
  - Each pixel's area = `(r2² - r1²) × Δθ / 2`
  - Where `r1` and `r2` are the inner and outer radii of the pixel
  - And `Δθ` is the angular width of the pixel

- **Impact**: A 5 km² storm now requires the same physical area regardless of whether it's near the radar center or at the edge of coverage, ensuring consistent storm detection across the entire radar domain.

## Displacement-Based New Storm Detection

The new storm detection now incorporates **displacement caused by wind/advection** using **patch-based cross-correlation** to improve accuracy:

### **Problem**
Traditional overlap-based detection can incorrectly classify moving storms as "new" storms, leading to false positives.

### **Solution: Patch-Based Cross-Correlation**
The displacement-based detection system:
1. **Divides frames into patches** (default: 64×64 pixels) for local displacement analysis
2. **Computes cross-correlation** for each patch to estimate local displacement caused by wind/advection
3. **Creates displacement field** by interpolating patch displacement vectors
4. **Predicts storm positions** using local displacement vectors at each pixel
5. **Compares current storms** with predicted positions instead of just previous positions

### **Important Note**
The displacement vectors represent **pixel displacement caused by wind/advection**, not actual wind speed or direction. They are used solely for storm tracking and position prediction, not for meteorological wind analysis.

### **Implementation**
- `compute_displacement_vectors()`: Uses **patch-based cross-correlation** to estimate displacement caused by wind/advection
- `create_displacement_field()`: Interpolates patch displacements into continuous displacement field
- `predict_storm_positions()`: Moves previous storms according to **local displacement field**
- `detect_new_storm_formations()`: Uses predicted positions for overlap comparison

### **Benefits** 
- **Spatial Accuracy**: Captures local displacement patterns and variations
- **Better Storm Tracking**: More accurate prediction of storm movement
- **Reduced False Positives**: Only storms that don't match predicted positions are classified as "new"
- **Similar to Paper**: Implementation follows the approach in the referenced MDPI paper
- **Configurable**: Can be disabled with `use_displacement_prediction=False` for comparison

### **Parameters**
- `patch_size`: Size of patches for cross-correlation (default: 64)
- `patch_stride`: Stride between patches (default: 32)
- `max_displacement`: Maximum expected displacement in pixels (default: 20)

## High-Reflectivity Patch Selection

The displacement computation now includes **high-reflectivity patch selection** to focus on physically meaningful regions and improve computational efficiency:

### **Problem**
Computing displacement vectors on all patches (including low-reflectivity background areas) is:
- **Computationally expensive**: Many patches with little useful information
- **Physically meaningless**: Background areas don't provide reliable displacement information
- **Visually cluttered**: Too many arrows in animation

### **Solution: Training-Style Patch Selection**
The system now uses the **same patch selection logic as the training scripts**:
1. **Normalize patches** using the same normalization as training (`patch / maxv`)
2. **Count high-reflectivity pixels** above `patch_thresh` (default: 0.35 normalized)
3. **Select patches** where fraction of high-reflectivity pixels ≥ `patch_frac` (default: 0.025)
4. **Compute displacement** only on selected patches
5. **Visualize arrows** only at selected patch centers

### **Benefits**
- **Computational Efficiency**: Significantly faster displacement computation
- **Physical Relevance**: Focus on storm regions where displacement matters
- **Cleaner Visualization**: Arrows only appear on meaningful storm regions
- **Consistency**: Same logic as training patch extraction
- **Configurable**: Can be disabled for comparison or adjusted for different datasets

### **Parameters**
- `patch_thresh`: Threshold for patch selection (default: 0.35, normalized)
- `patch_frac`: Minimum fraction of pixels above threshold (default: 0.025)
- `maxv`: Maximum value for normalization (default: 85.0)
- `use_high_reflectivity_patches`: Enable/disable patch selection (default: True)

### **CLI Usage**
```bash
python src/utils/storm_utils.py \
  --preds predictions/unet3dcnn_example/test_preds_dBZ.npy \
  --targets predictions/unet3dcnn_example/test_targets_dBZ.npy \
  --out experiments/runs/unet3dcnn_example/results/storm_eval.json \
  --reflectivity_threshold 45 \
  --area_threshold_km2 5.0 \
  --dilation_iterations 5 \
  --overlap_threshold 0.2 \
  --use_displacement_prediction \
  --patch_size 64 \
  --patch_stride 32 \
  --patch_thresh 0.35 \
  --patch_frac 0.025 \
  --maxv 85.0 \
  --use_high_reflectivity_patches
```

**Additional CLI Arguments:**
- `--patch_thresh`: Threshold for patch selection (default: 0.35, normalized)
- `--patch_frac`: Minimum fraction of pixels above threshold (default: 0.025)
- `--maxv`: Maximum value for normalization (default: 85.0)
- `--use_high_reflectivity_patches`: Use only patches with high reflectivity (default: True)
- `--no_high_reflectivity_patches`: Disable high-reflectivity patch selection (use all patches)

## CLI Usage: Evaluate Storm Initiation Predictions and Forecasting Metrics

After running testing and saving predictions/targets as `.npy` files, evaluate both storm initiations and forecasting metrics with:

```bash
python src/utils/storm_utils.py \
  --preds predictions/unet3dcnn_example/test_preds_dBZ.npy \
  --targets predictions/unet3dcnn_example/test_targets_dBZ.npy \
  --out experiments/runs/unet3dcnn_example/results/storm_eval.json \
  --reflectivity_threshold 45 \
  --area_threshold_km2 5.0 \
  --dilation_iterations 5 \
  --overlap_threshold 0.2 \
  --use_displacement_prediction \
  --patch_size 64 \
  --patch_stride 32
```

- `--preds`: Path to predicted reflectivity `.npy` file (shape: N, C, H, W or N, H, W)
- `--targets`: Path true reflectivity `.npy` file
- `--out`: Output JSON file for evaluation results
- `--reflectivity_threshold`: dBZ threshold for storm detection (default: 45)
- `--area_threshold_km2`: Minimum storm area in km² (default: 5.0)
- `--dilation_iterations`: Dilation iterations for storm region merging (default: 5)
- `--overlap_threshold`: Overlap ratio for matching storms (default: 0.2)
- `--use_displacement_prediction`: Enable displacement-based prediction for new storm detection (default: True)
- `--no_displacement_prediction`: Disable displacement-based prediction (use overlap-based method)

- `--patch_size`: Patch size for displacement computation (default: 64)
- `--patch_stride`: Patch stride for displacement computation (default: 32)

**Output includes both:**
- **Storm initiation metrics**: correct, early, late, false positives, etc.
- **Forecasting metrics**: B-MSE, CSI, HSS for thresholds [2, 5, 10, 30, 45] dBZ



## CLI Usage: Count Storms by Data Sections

Analyze storm and new storm counts across temporal sections of radar data:

```bash
python src/utils/storm_section_counter.py \
  --npy_path data/processed/ZH_radar_dataset.npy \
  --interval_percent 5 \
  --batch_size 10 \
  --reflectivity_threshold 45 \
  --area_threshold_km2 10.0 \
  --dilation_iterations 5 \
  --overlap_threshold 0.1 \
  --out results/storm_section_counts.json
```

- `--npy_path`: Path to radar data `.npy` file (shape: T, H, W or N, C, H, W)
- `--interval_percent`: Section size as percentage of total data length (default: 5)
- `--batch_size`: Number of frames to process at once (for memory efficiency) (default: 10)
- `--reflectivity_threshold`: dBZ threshold for storm detection (default: 45)
- `--area_threshold_km2`: Minimum storm area in km² (default: 5.0)
- `--dilation_iterations`: Dilation iterations for storm region merging (default: 5)
- `--overlap_threshold`: Overlap threshold for new storm detection (default: 0.1)
- `--out`: Optional output JSON file for results

## Main Functions

### In `storm_utils.py` (detection & evaluation)
- `detect_storms(data, reflectivity_threshold, area_threshold_km2, dilation_iterations)`
  - Detects storms in each frame of radar data using physical area calculations for polar coordinates.
- `detect_new_storm_formations(data, reflectivity_threshold, area_threshold_km2, dilation_iterations, overlap_threshold, use_displacement_prediction=True, patch_size=64, patch_stride=32, patch_thresh=0.35, patch_frac=0.025, maxv=85.0, use_high_reflectivity_patches=True)`
  - Identifies new storm initiations over time using displacement-based prediction to account for storm movement.
  - Includes high-reflectivity patch selection for displacement computation.
- `compute_displacement_vectors(data, patch_size, patch_stride, max_displacement, show_progress=True, patch_thresh=0.35, patch_frac=0.025, maxv=85.0, use_high_reflectivity_patches=True)`
  - Computes displacement vectors using **patch-based cross-correlation** between consecutive radar frames.
  - Divides each frame into patches and computes cross-correlation for each patch.
  - Uses high-reflectivity patch selection to focus on storm regions.
  - Returns both global displacement vectors, spatial displacement fields, and selected patch centers.
- `create_displacement_field(patch_displacements, patch_positions, field_shape)`
  - Creates a displacement field by interpolating patch displacement vectors using inverse distance weighting.
- `predict_storm_positions(previous_storms, displacement_field, data_shape)`
  - Predicts where storms from the previous frame should be in the current frame based on **local displacement field**.
  - Each storm pixel moves according to the local displacement vector at that location.
- `evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold)`
  - Compares predicted and true new storm initiations - returns metrics.
- `count_storms_by_section(data, interval_percent, batch_size, ...)`
  - Count storms and new storms in temporal sections of data.
- `compute_csi_hss(pred, target, threshold)`
  - Computes CSI (Critical Success Index) and HSS (Heidke Skill Score) for a given threshold.
- `compute_b_mse(pred, target, maxv, eps)`
  - Computes B-MSE (Balanced Mean Squared Error) using the same weighting scheme as training.
- `compute_forecasting_metrics(pred, target, maxv, eps)`
  - Computes comprehensive forecasting metrics including CSI, HSS, and B-MSE for all thresholds.

### In `storm_animation_utils.py` (visualization)
- `animate_storms(data, ...)`
  - Create matplotlib animation of storms over time.
- `animate_new_storms(data, new_storms_result)`
  - Animate new storm initiations over time.
- `animate_storms_polar(data, ...)` and `animate_storms_polar_comparison(true_data, pred_data, ...)`
  - Polar coordinate visualizations for radar data.
- `animate_new_storms_with_wind(data, reflectivity_threshold, area_threshold_km2, dilation_iterations, overlap_threshold, interval, patch_size=64, patch_stride=32, patch_thresh=0.35, patch_frac=0.025, maxv=85.0, use_high_reflectivity_patches=True)`
  - Animate displacement-based new storm detection, showing current storms (red), predicted positions (orange dashed), new storms (lime), and displacement vectors (red arrows).
  - Displacement arrows are only shown on high-reflectivity patches for cleaner visualization.

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

# Evaluate storm initiation predictions
storm_metrics = storm_utils.evaluate_new_storm_predictions(pred_new_storms, target_new_storms)
print("Storm initiation metrics:", storm_metrics)

# Compute forecasting metrics (CSI, HSS, B-MSE)
# Note: predictions and targets are already in dBZ
forecasting_metrics = storm_utils.compute_forecasting_metrics(preds, targets)
print("B-MSE:", forecasting_metrics['b_mse'])
print("CSI by threshold:", forecasting_metrics['csi_by_threshold'])
print("HSS by threshold:", forecasting_metrics['hss_by_threshold'])

# Animate storms (in a notebook)
ani = storm_animation_utils.animate_storms(preds)
from IPython.display import HTML
display(HTML(ani.to_jshtml()))
```
