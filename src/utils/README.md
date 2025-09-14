# Utils

This folder contains utility functions for storm detection, evaluation, and visualization in radar nowcasting experiments.

## Available Scripts

- **storm_utils.py** — Storm detection and evaluation functions
- **storm_animation_utils.py** — Animation and visualization functions

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
- Storm initiation metrics: correct (predicted at correct time step), early (predicted 1 time step early), late (predicted 1 time step late), incorrect initiations, etc.
- Forecasting metrics: Balanced Mean Squared Error (B-MSE), Critical Success Index (CSI), Heidke Skill Score (HSS) for thresholds [2, 5, 10, 30, 45] dBZ

**Note**: The forecasting metrics (B-MSE, CSI, HSS) computed by `storm_utils.py` is done on the predicted and true data arrays from testing. These arrays are the Composite Reflectivity (Maximum Intensity Projection over altitude).

## Animation & Visualization

See `notebooks/storm_animation.ipynb` for examples of how to use the animation functions.

## Storm Detection and Initiation Features

- **Physical Area Calculations**: Storm detection accounts for polar coordinate geometry to calculate the area of storms in km². In radar polar coordinates, storms with the same number of pixels represent different physical areas depending on their distance from the radar. A storm near the radar center covers a smaller physical area than a storm with the same pixel count at the edge measurements. Physical area calculations thus ensures consistent storm detection across the entire radar domain.

- **Displacement-Based Tracking**: Uses patch-based cross-correlation to track storm movement caused by wind. Strong winds can move storms significantly between time steps, making them appear as "new" storms when using simple overlap tracking. Our enhanced displacement tracking divides radar frames into overlapping patches and uses cross-correlation to find how each patch moves between consecutive frames. Only patches with sufficient reflectivity and good correlation quality is used. It also uses temporal smoothing with moving average (70% current + 30% previous displacement) to prevent sudden wind direction changes and maintain temporal continuity. The system filters out patches with low correlation quality (default threshold: 0.5) to ensure only reliable displacement measurements are used, then creates dense displacement fields by interpolating sparse patch displacements using distance-weighted interpolation where closer patches have higher influence on the interpolated displacement at each pixel. When no nearby patches exist for interpolation, it uses previous displacement field as fallback to, and applies light Gaussian smoothing to reduce interpolation noise. These improvements predict where storms should be based on wind movement more accurately than traditional methods, reducing false positive new storm detections.

- **Azimuth Wrapping for Storm Tracking**: An issue in radar storm tracking occurs in the Cartesian coordiate system when storms move through the radar center, crossing the 0°/360° azimuth boundary. The Cartesian coordinate approach which we use fails because this treat azimuth 359° and 0° as opposite ends of the frame, while they are actually adjacent in the circular radar coordinate system. This causes incorrect overlap calculations and displacement tracking, leading to false "new storm" detections when storms move through the center. To address this we use an azimuth wrapping approach: we extend the radar data by duplicating the first 100 azimuth bins (0°-100°) at the end (after 360°), creating a continuous dataset that handles the circular nature of azimuth. All displacement and overlap calculations are performed in this wrapped space, then coordinates are mapped back to the original data space. Thus storm tracking is correct also when storms move through the radar center, eliminating false new storm detections caused by coordinate system artifacts.

See the main [README.md](../../README.md) for the full pipeline.
