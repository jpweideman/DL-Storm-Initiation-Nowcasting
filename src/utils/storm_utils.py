import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from matplotlib.path import Path
import argparse
import json
from tqdm import tqdm
import os

def compute_csi_hss(pred, target, threshold):
    """
    Compute CSI (Critical Success Index) and HSS (Heidke Skill Score) for a given threshold.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values in dBZ.
    target : np.ndarray
        Ground truth values in dBZ.
    threshold : float
        Threshold in dBZ to convert to binary (0/1).
        
    Returns
    -------
    tuple
        (CSI, HSS) scores.
    """
    # Convert to binary using threshold
    pred_binary = (pred >= threshold).astype(int)
    target_binary = (target >= threshold).astype(int)
    
    # Calculate confusion matrix elements
    TP = np.sum((pred_binary == 1) & (target_binary == 1))  # True Positive
    FN = np.sum((pred_binary == 0) & (target_binary == 1))  # False Negative
    FP = np.sum((pred_binary == 1) & (target_binary == 0))  # False Positive
    TN = np.sum((pred_binary == 0) & (target_binary == 0))  # True Negative
    
    # Calculate CSI
    if TP + FN + FP == 0:
        CSI = 0.0
    else:
        CSI = TP / (TP + FN + FP)
    
    # Calculate HSS
    denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    if denominator == 0:
        HSS = 0.0
    else:
        HSS = (TP * TN - FN * FP) / denominator
    
    return CSI, HSS

def compute_b_mse(pred, target):
    """
    Compute B-MSE (Balanced Mean Squared Error) using the same weighting scheme as in training.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values in dBZ.
    target : np.ndarray
        Ground truth values in dBZ.
        
    Returns
    -------
    float
        B-MSE value.
    """
    # Compute weights based on dBZ values directly
    w = np.ones_like(target)
    w = np.where(target < 2, 1.0, w)
    w = np.where((target >= 2) & (target < 5), 2.0, w)
    w = np.where((target >= 5) & (target < 10), 5.0, w)
    w = np.where((target >= 10) & (target < 30), 10.0, w)
    w = np.where((target >= 30) & (target < 45), 30.0, w)
    w = np.where(target >= 45, 45.0, w)
    
    # B-MSE
    b_mse = np.sum(w * (pred - target) ** 2) / np.sum(w)
    return b_mse

def compute_forecasting_metrics(pred, target):
    """
    Compute comprehensive forecasting metrics including CSI, HSS, and B-MSE.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values in dBZ.
    target : np.ndarray
        Ground truth values in dBZ.
        
    Returns
    -------
    dict
        Dictionary containing all metrics:
        - 'b_mse': Balanced Mean Squared Error
        - 'csi_by_threshold': CSI scores for thresholds [2, 5, 10, 30, 45] dBZ
        - 'hss_by_threshold': HSS scores for thresholds [2, 5, 10, 30, 45] dBZ
    """
    # Compute B-MSE
    b_mse_value = compute_b_mse(pred, target)
    
    # Compute CSI and HSS for different thresholds
    thresholds = [2, 5, 10, 30, 45]
    csi_by_threshold = {}
    hss_by_threshold = {}
    
    for th in thresholds:
        csi, hss = compute_csi_hss(pred, target, th)
        csi_by_threshold[f"csi_{th}"] = float(csi)  # Convert to Python float
        hss_by_threshold[f"hss_{th}"] = float(hss)  # Convert to Python float
    
    return {
        "b_mse": float(b_mse_value),  # Convert to Python float
        "csi_by_threshold": csi_by_threshold,
        "hss_by_threshold": hss_by_threshold
    }

def compute_polar_pixel_areas(shape, pixel_spacing_km=0.5):
    """
    Compute the physical area (in km²) of each pixel in polar radar coordinates.
    
    Parameters
    ----------
    shape : tuple
        Shape of the radar data (azimuth_bins, range_bins) where:
        - azimuth_bins: number of azimuth angles (typically 360)
        - range_bins: number of range bins at pixel_spacing_km intervals
    pixel_spacing_km : float
        Distance between pixels in km (default: 0.5 km)
        
    Returns
    -------
    np.ndarray
        Array of shape (azimuth_bins, range_bins) containing the area of each pixel in km²
    """
    azimuth_bins, range_bins = shape
    
    # Create range and azimuth arrays
    ranges = np.arange(range_bins) * pixel_spacing_km  # Distance from radar in km
    azimuths = np.arange(azimuth_bins) * (360 / azimuth_bins)  # Azimuth angles in degrees
    
    # Convert azimuths to radians
    azimuths_rad = np.radians(azimuths)
    
    # Create meshgrid - CORRECTED: azimuth first, range second
    A, R = np.meshgrid(azimuths_rad, ranges, indexing='ij')  # (azimuth_bins, range_bins)
    
    # Compute pixel areas using polar geometry
    # Area = (r2² - r1²) * Δθ / 2
    # where r1 and r2 are the inner and outer radii of the pixel
    # and Δθ is the angular width of the pixel
    
    # For each range bin, compute the area
    areas = np.zeros_like(R, dtype=float)
    
    for j in range(range_bins):
        r1 = j * pixel_spacing_km
        r2 = (j + 1) * pixel_spacing_km
        
        # Angular width of each pixel (in radians)
        delta_theta = 2 * np.pi / azimuth_bins
        
        # Area of this pixel
        pixel_area = (r2**2 - r1**2) * delta_theta / 2
        
        areas[:, j] = pixel_area  # All azimuths at this range have same area
    
    return areas

def detect_storms(data, reflectivity_threshold=45, area_threshold_km2=5.0, dilation_iterations=5):
    """
    Detects storms in radar data using physical area calculations for polar coordinates.

    Parameters:
    - data: ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - reflectivity_threshold: threshold for storm detection (dBZ)
    - area_threshold_km2: minimum storm area in km² (default: 5.0 km²)
    - dilation_iterations: dilation iterations for storm smoothing

    Returns:
    - List of dictionaries containing storm count and coordinates per frame.
    """
    # Compute pixel areas for the radar geometry
    pixel_areas = compute_polar_pixel_areas(data.shape[1:])
    
    results = []

    for t in range(data.shape[0]):
        frame_data = data[t]
        mask = frame_data > reflectivity_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
        contours = find_contours(dilated_mask.astype(float), 0.5)

        storms_in_frame = []

        for contour in contours:
            path = Path(contour[:, ::-1])  # (x, y) ordering

            # Grid coordinates - for (azimuth_bins, range_bins) shape
            # frame_data.shape = (azimuth_bins, range_bins)
            x_grid, y_grid = np.meshgrid(
                np.arange(frame_data.shape[1]),  # range_bins (x-axis)
                np.arange(frame_data.shape[0])   # azimuth_bins (y-axis)
            )
            coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
            inside = path.contains_points(coords).reshape(frame_data.shape)  # (azimuth_bins, range_bins)

            # Calculate physical area using pixel areas
            storm_pixels = mask & inside
            physical_area = np.sum(pixel_areas * storm_pixels)

            if physical_area >= area_threshold_km2:
                storm_coords = contour[:, [1, 0]].tolist()  # (x, y) points
                storms_in_frame.append({
                    "mask": inside.astype(int), 
                    "contour": storm_coords,
                    "physical_area_km2": float(physical_area)
                })

        frame_result = {
            "time_step": t,
            "storm_count": len(storms_in_frame),
            "storm_coordinates": [s["contour"] for s in storms_in_frame],
            "storm_masks": [s["mask"] for s in storms_in_frame],
            "storm_areas_km2": [s["physical_area_km2"] for s in storms_in_frame],
        }

        results.append(frame_result)

    return results

def detect_new_storm_formations(data, reflectivity_threshold=45, area_threshold_km2=5.0, dilation_iterations=5, overlap_threshold=0.1):
    """
    Detects new storm formations: frames, count of newly formed storms, and their coordinates.

    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - reflectivity_threshold: float, reflectivity threshold (dBZ) to identify storms
    - area_threshold_km2: float, minimum storm area in km² for a region to be considered a storm
    - dilation_iterations: int, number of binary dilation iterations to smooth/merge storms
    - overlap_threshold: float, maximum allowed overlap between a current storm and any previous storm for it to be considered "new".

    Returns:
    - new_storms_summary: list of dict
        Each dictionary contains:
            - 'time_step': int
            - 'new_storm_count': int
            - 'new_storm_coordinates': list of storm contour coordinates (list of (x, y))

    """
    storm_results = detect_storms(data, reflectivity_threshold, area_threshold_km2, dilation_iterations)
    new_storms_summary = []

    previous_masks = []

    for frame_result in storm_results:
        new_count = 0
        new_coords = []
        current_masks = frame_result['storm_masks']
        current_coords = frame_result['storm_coordinates']

        for i, current in enumerate(current_masks):
            is_new = True
            for prev in previous_masks:
                overlap = np.sum(current & prev) / np.sum(current)
                if overlap > overlap_threshold:
                    is_new = False
                    break
            if is_new:
                new_count += 1
                new_coords.append(current_coords[i])

        new_storms_summary.append({
            "time_step": frame_result['time_step'],
            "new_storm_count": new_count,
            "new_storm_coordinates": new_coords
        })

        previous_masks = current_masks

    return new_storms_summary

def evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold=0.2):
    """
    Compares predicted and true new storm initiations.

    Parameters:
    - new_storms_pred: list of dicts from detect_new_storm_formations(pred_max_cappi)
    - new_storms_true: list of dicts from detect_new_storm_formations(true_max_cappi)
    - overlap_threshold: float, minimum overlap ratio to count as a match

    Returns:
    - dict with counts:
        {
            "correct": int,  # matched at same time
            "early": int,    # matched one time step before
            "late": int,     # matched one time step after
            "false_positives": int,  # predicted storms with no match in ±1 time step
            "total_true": int,  # total true new storms
            "total_pred": int,  # total predicted new storms
            "correct_over_true": float,
            "correct_over_pred": float,
            "anytime_ratio": float,
            "false_positive_ratio": float,
        }
    """
    # Helper to get mask for a storm contour
    def contour_to_mask(contour, shape):
        from matplotlib.path import Path as MplPath
        # Ensure shape is a 2-tuple
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif len(shape) == 0:
            shape = (1, 1)
        xg, yg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        path = MplPath(np.array(contour))
        mask = path.contains_points(coords).reshape(shape)
        return mask

    # Build lookup: time_step -> list of (mask, used_flag)
    def build_storm_masks(new_storms, data_shape):
        lookup = {}
        for entry in new_storms:
            t = entry["time_step"]
            masks = []
            for contour in entry["new_storm_coordinates"]:
                mask = contour_to_mask(contour, data_shape)
                masks.append({"mask": mask, "used": False})
            lookup[t] = masks
        return lookup

    # infer data_shape from all contours in true and pred
    max_x, max_y = 0, 0
    found = False
    for storms in (new_storms_true, new_storms_pred):
        for entry in storms:
            for contour in entry["new_storm_coordinates"]:
                arr = np.array(contour)
                if arr.size == 0:
                    continue
                found = True
                if arr.ndim == 1 and arr.shape[0] == 2:
                    # Single point
                    x, y = arr[0], arr[1]
                    max_x = max(max_x, int(x))
                    max_y = max(max_y, int(y))
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    # Contour
                    max_x = max(max_x, int(arr[:, 0].max()))
                    max_y = max(max_y, int(arr[:, 1].max()))
    if not found:
        return {"correct": 0, "early": 0, "late": 0, "false_positives": sum(e["new_storm_count"] for e in new_storms_pred), "total_true": 0, "total_pred": sum(e["new_storm_count"] for e in new_storms_pred), "correct_over_true": 0.0, "correct_over_pred": 0.0, "anytime_ratio": 0.0, "false_positive_ratio": 0.0}
    data_shape = (max_y + 1, max_x + 1)

    # Build mask lookups
    pred_lookup = build_storm_masks(new_storms_pred, data_shape)
    true_lookup = build_storm_masks(new_storms_true, data_shape)

    correct = 0
    early = 0
    late = 0
    matched_pred = set()  # (t, idx)
    matched_true = set()  # (t, idx)

    # For each true storm, look for matching pred storm in t-1, t, t+1
    print('Evaluating (true storms)...')
    for t, true_storms in true_lookup.items():
        for i, true_storm in enumerate(true_storms):
            found = False
            for dt, label in zip([0, -1, 1], ["correct", "early", "late"]):
                tt = t + dt
                if tt in pred_lookup:
                    for j, pred_storm in enumerate(pred_lookup[tt]):
                        if (tt, j) in matched_pred:
                            continue
                        # Compute overlap: intersection / area of true storm
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        area_true = np.sum(true_storm["mask"])
                        if area_true == 0:
                            continue
                        overlap = intersection / area_true
                        if overlap >= overlap_threshold:
                            if label == "correct":
                                correct += 1
                            elif label == "early":
                                early += 1
                            elif label == "late":
                                late += 1
                            matched_pred.add((tt, j))
                            matched_true.add((t, i))
                            found = True
                            break
                if found:
                    break

    # False positives: predicted storms not matched to any true storm in ±1 time step
    print('Evaluating (false positives)...')
    false_positives = 0
    for t, pred_storms in pred_lookup.items():
        for j, pred_storm in enumerate(pred_storms):
            if (t, j) in matched_pred:
                continue
            # Check if this pred storm matches any true storm in t-1, t, t+1
            matched = False
            for dt in [-1, 0, 1]:
                tt = t + dt
                if tt in true_lookup:
                    for i, true_storm in enumerate(true_lookup[tt]):
                        if (tt, i) in matched_true:
                            continue
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        area_pred = np.sum(pred_storm["mask"])
                        if area_pred == 0:
                            continue
                        overlap = intersection / area_pred
                        if overlap >= overlap_threshold:
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                false_positives += 1

    total_true = sum(len(v) for v in true_lookup.values())
    total_pred = sum(len(v) for v in pred_lookup.values())

    # Ratios
    correct_over_true = correct / total_true if total_true > 0 else 0.0
    correct_over_pred = correct / total_pred if total_pred > 0 else 0.0
    anytime_ratio = (correct + early + late) / total_true if total_true > 0 else 0.0
    false_positive_ratio = false_positives / total_pred if total_pred > 0 else 0.0

    return {
        "correct": int(correct),
        "early": int(early),
        "late": int(late),
        "false_positives": int(false_positives),
        "total_true": int(total_true),
        "total_pred": int(total_pred),
        "correct_over_true": float(correct_over_true),
        "correct_over_pred": float(correct_over_pred),
        "anytime_ratio": float(anytime_ratio),
        "false_positive_ratio": float(false_positive_ratio),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Evaluate new storm predictions from .npy files and save results.
        This script loads prediction and target arrays (N, C, H, W),
        reduces them to (T, H, W) by taking the max over the channel dimension (axis=1, max CAPPI),
        then runs detect_new_storm_formations and evaluate_new_storm_predictions.
        Results are printed and saved as JSON.
        """
    )
    parser.add_argument('--preds', type=str, required=True, help='Path to predicted storms .npy file (shape: N, C, H, W or N, H, W)')
    parser.add_argument('--targets', type=str, required=True, help='Path to true storms .npy file (shape: N, C, H, W or N, H, W)')
    parser.add_argument('--out', type=str, required=True, help='Output JSON file for evaluation results')
    parser.add_argument('--overlap_threshold', type=float, default=0.2, help='Overlap threshold for matching storms (default: 0.2)')
    parser.add_argument('--reflectivity_threshold', type=float, default=45, help='Reflectivity threshold for storm detection (default: 45)')
    parser.add_argument('--area_threshold_km2', type=float, default=5.0, help='Area threshold for storm detection in km² (default: 5.0)')
    parser.add_argument('--dilation_iterations', type=int, default=5, help='Dilation iterations for storm detection (default: 5)')

    args = parser.parse_args()

    # Load shape and dtype from metadata files if they exist
    def load_memmap_with_meta(array_path):
        meta_path = array_path.replace('.npy', '_meta.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            shape = tuple(meta['shape'])
            dtype = str(meta['dtype'])
            return np.memmap(array_path, dtype=dtype, mode='r', shape=shape)
        else:
            # Fallback: try np.load (for legacy files)
            return np.load(array_path, mmap_mode='r')

    pred = load_memmap_with_meta(args.preds)
    tgt = load_memmap_with_meta(args.targets)

    # Always reduce to (T, H, W) by taking max over channel dimension if present
    # Documented: This is CAPPI (Constant Altitude Plan Position Indicator)
    # If input is (N, C, H, W), take max over axis=1
    # If input is already (N, H, W), do nothing
    if pred.ndim == 4:
        pred_cappi = np.max(pred, axis=1)
        tgt_cappi = np.max(tgt, axis=1)
    elif pred.ndim == 3:
        pred_cappi = pred
        tgt_cappi = tgt
    else:
        raise ValueError("Input arrays must be of shape (N, C, H, W) or (N, H, W)")

    # Progress bar for storm detection
    def detect_with_progress(data, **kwargs):
        storms = []
        for t in tqdm(range(data.shape[0]), desc=kwargs.get('desc', 'Detecting storms')):
            storms.append(detect_storms(data[t:t+1], **{k: v for k, v in kwargs.items() if k != 'desc'})[0])
        # Reformat to match detect_new_storm_formations output
        # (detect_new_storm_formations returns a summary, not raw storms)
        # So we call the original function after progress for correct output
        return detect_new_storm_formations(data, **{k: v for k, v in kwargs.items() if k != 'desc'})

    print('Detecting new storm formations in predictions...')
    pred_storms = detect_with_progress(
        pred_cappi,
        reflectivity_threshold=args.reflectivity_threshold,
        area_threshold_km2=args.area_threshold_km2,
        dilation_iterations=args.dilation_iterations,
        desc='Pred storms')
    print('Detecting new storm formations in targets...')
    tgt_storms = detect_with_progress(
        tgt_cappi,
        reflectivity_threshold=args.reflectivity_threshold,
        area_threshold_km2=args.area_threshold_km2,
        dilation_iterations=args.dilation_iterations,
        desc='True storms')

    # Evaluate storm initiation predictions
    storm_results = evaluate_new_storm_predictions(pred_storms, tgt_storms, overlap_threshold=args.overlap_threshold)
    
    # Compute forecasting metrics (CSI, HSS, B-MSE)
    forecasting_metrics = compute_forecasting_metrics(pred_cappi, tgt_cappi)
    
    # Combine results
    results = {
        "storm_initiation_metrics": storm_results,
        "forecasting_metrics": forecasting_metrics
    }
    
    print("\n=== STORM INITIATION METRICS ===")
    print(json.dumps(storm_results, indent=2))
    
    print("\n=== FORECASTING METRICS ===")
    print(f"B-MSE: {forecasting_metrics['b_mse']:.4f}")
    print("CSI by threshold:")
    for th, csi in forecasting_metrics['csi_by_threshold'].items():
        print(f"  {th}: {csi:.4f}")
    print("HSS by threshold:")
    for th, hss in forecasting_metrics['hss_by_threshold'].items():
        print(f"  {th}: {hss:.4f}")

    # Save to JSON
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

