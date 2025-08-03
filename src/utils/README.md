# Utils

This folder contains utility functions for storm detection, evaluation, and visualization in radar nowcasting experiments.

## Available Scripts

- **storm_utils.py** — Storm detection and evaluation functions
- **storm_animation_utils.py** — Animation and visualization functions  
- **storm_section_counter.py** — Count storms in temporal sections of data

## Storm Detection & Evaluation

### Evaluate Storm Initiation Predictions

After running testing and saving predictions/targets as `.npy` files, evaluate storm initiations:

```bash
python src/utils/storm_utils.py \
  --preds predictions/unet3dcnn_example/test_preds_dBZ.npy \
  --targets predictions/unet3dcnn_example/test_targets_dBZ.npy \
  --out experiments/runs/unet3dcnn_example/results/storm_eval.json \
  --reflectivity_threshold 45 \
  --area_threshold_km2 10.0 \
  --dilation_iterations 5 \
  --overlap_threshold 0.2 \
  --storm_tracking_overlap_threshold 0.2 \
  --use_displacement_prediction \
  --patch_size 32 \
  --patch_stride 16 \
  --patch_thresh 35 \
  --patch_frac 0.015 \
  --maxv 85.0
```

**Output includes:**
- Storm initiation metrics: correct, early, late, false positives, etc.
- Forecasting metrics: B-MSE, CSI, HSS for thresholds [2, 5, 10, 30, 45] dBZ

### Count Storms by Data Sections

Analyze storm and new storm counts across temporal sections:

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

## Animation & Visualization

See the `notebooks/` folder for examples of how to use the animation functions.

## Storm Detection and Initiation Features

- **Physical Area Calculations**: Storm detection accounts for polar coordinate geometry to calculate the area of storms in km². In radar polar coordinates, the same number of pixels represents different physical areas depending on distance from the radar. A storm near the radar center covers a smaller physical area than the same storm at the edge of coverage. Physical area calculations thus ensures consistent storm detection across the entire radar domain.

- **Displacement-Based Tracking**: Uses patch-based cross-correlation to track storm movement caused by wind. Strong winds can move storms significantly between time steps, making them appear as "new" storms when using simple overlap tracking. Displacement-based tracking predicts where storms should be based on wind movement, reducing false positive new storm detections.

See the main [README.md](../../README.md) for the full pipeline.
