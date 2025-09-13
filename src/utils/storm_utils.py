import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from matplotlib.path import Path
import argparse
import json
from tqdm import tqdm
import os
from scipy.signal import correlate2d
from scipy.optimize import minimize_scalar

def compute_csi_hss(pred, target, threshold):
    """
    Compute Critical Success Index (CSI) and Heidke Skill Score (HSS) for a given threshold.
    
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
    # Convert to binary
    pred_binary = (pred >= threshold).astype(int)
    target_binary = (target >= threshold).astype(int)
    
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    
    if TP + FN + FP == 0:
        CSI = 0.0
    else:
        CSI = TP / (TP + FN + FP)
    
    denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    if denominator == 0:
        HSS = 0.0
    else:
        HSS = (TP * TN - FN * FP) / denominator
    
    return CSI, HSS

def compute_b_mse(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute Balanced Mean Squared Error (B-MSE) using the same weighting scheme as in training.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values in dBZ.
    target : np.ndarray
        Ground truth values in dBZ.
    maxv : float, optional
        Maximum value for normalization (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).
    
    Returns
    -------
    float
        B-MSE value.
    """
    if pred.max() <= 1.0 and target.max() <= 1.0:
        pred = pred * (maxv + eps)
        target = target * (maxv + eps)
    
    w = np.ones_like(target, dtype=np.float32)  

    w = np.where(target < 2, 1.0, w)
    w = np.where((target >= 2) & (target < 5), 2.0, w)
    w = np.where((target >= 5) & (target < 10), 5.0, w)
    w = np.where((target >= 10) & (target < 30), 10.0, w)
    w = np.where((target >= 30) & (target < 45), 30.0, w)
    w = np.where(target >= 45, 45.0, w)
    
    b_mse = np.mean(w * (pred - target) ** 2)
    return b_mse

def compute_forecasting_metrics(pred, target, maxv=85.0, eps=1e-6):
    """
    Compute forecasting metrics including CSI, HSS, and B-MSE.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values in dBZ.
    target : np.ndarray
        Ground truth values in dBZ.
    maxv : float, optional
        Maximum value for normalization (default: 85.0).
    eps : float, optional
        Small epsilon to avoid division by zero (default: 1e-6).
    
    Returns
    -------
    dict
        Dictionary containing all metrics:
        - 'b_mse': Balanced Mean Squared Error
        - 'csi_by_threshold': CSI scores for thresholds [2, 5, 10, 30, 45] dBZ
        - 'hss_by_threshold': HSS scores for thresholds [2, 5, 10, 30, 45] dBZ
    """

    pred_dBZ = pred.copy()
    target_dBZ = target.copy()
    if pred.max() <= 1.0 and target.max() <= 1.0:
        pred_dBZ = pred * (maxv + eps)
        target_dBZ = target * (maxv + eps)
    
    # B-MSE
    b_mse_value = compute_b_mse(pred, target, maxv=maxv, eps=eps)
    
    # CSI and HSS thresholds
    thresholds = [2, 5, 10, 30, 45]
    csi_by_threshold = {}
    hss_by_threshold = {}
    
    for th in thresholds:
        csi, hss = compute_csi_hss(pred_dBZ, target_dBZ, th)
        csi_by_threshold[f"csi_{th}"] = float(csi)  
        hss_by_threshold[f"hss_{th}"] = float(hss)  
    
    return {
        "b_mse": float(b_mse_value),  
        "csi_by_threshold": csi_by_threshold,
        "hss_by_threshold": hss_by_threshold
    }

def compute_polar_pixel_areas(shape, pixel_spacing_km=0.5):
    """
    Compute the physical area (in km^2) of each pixel in polar radar coordinates.
    
    Parameters
    ----------
    shape : tuple
        Shape of the radar data (azimuth_bins, range_bins) where:
        - azimuth_bins: number of azimuth angles (typically 360)
        - range_bins: number of range bins at pixel_spacing_km intervals
    pixel_spacing_km : float, optional
        Distance between pixels in km (default: 0.5).
    
    Returns
    -------
    np.ndarray
        Array of shape (azimuth_bins, range_bins) containing the area of each pixel in km^2.
    """
    azimuth_bins, range_bins = shape
    
    # Create range and azimuth arrays
    ranges = np.arange(range_bins) * pixel_spacing_km
    azimuths = np.arange(azimuth_bins) * (360 / azimuth_bins)
    
    # Convert to radians
    azimuths_rad = np.radians(azimuths)
    
    # Create meshgrid
    A, R = np.meshgrid(azimuths_rad, ranges, indexing='ij')
    
    # Compute pixel areas using polar geometry
    # Area = (r2² - r1²) * Δθ / 2
    # where r1 and r2 are the inner and outer radii of the pixel
    # and Δθ is the angular width of the pixel
    
    # Calculate area for each range bin
    areas = np.zeros_like(R, dtype=float)
    
    for j in range(range_bins):
        r1 = j * pixel_spacing_km
        r2 = (j + 1) * pixel_spacing_km
        
        # Angular width in radians
        delta_theta = 2 * np.pi / azimuth_bins
        
        pixel_area = (r2**2 - r1**2) * delta_theta / 2
        
        areas[:, j] = pixel_area
    
    return areas

def detect_storms(data, reflectivity_threshold=45, area_threshold_km2=10.0, dilation_iterations=5):
    """
    Detects storms in radar data using physical area calculations for polar coordinates.

    Parameters
    ----------
    data : np.ndarray
        Radar data of shape (T, H, W) where H=azimuth_bins, W=range_bins.
    reflectivity_threshold : float, optional
        Threshold for storm detection in dBZ (default: 45).
    area_threshold_km2 : float, optional
        Minimum storm area in km² (default: 10.0).
    dilation_iterations : int, optional
        Dilation iterations for storm smoothing (default: 5).

    Returns
    -------
    list
        List of dictionaries containing storm count and coordinates per frame.
    """
    # Get pixel areas for radar geometry
    pixel_areas = compute_polar_pixel_areas(data.shape[1:])
    
    results = []

    for t in range(data.shape[0]):
        frame_data = data[t]
        mask = frame_data > reflectivity_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
        contours = find_contours(dilated_mask.astype(float), 0.5)

        storms_in_frame = []

        for contour in contours:
            path = Path(contour[:, ::-1])  

            # Create grid coordinates
            x_grid, y_grid = np.meshgrid(
                np.arange(frame_data.shape[1]), 
                np.arange(frame_data.shape[0])   
            )
            coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
            inside = path.contains_points(coords).reshape(frame_data.shape)

            # Calculate physical area
            storm_pixels = mask & inside
            physical_area = np.sum(pixel_areas * storm_pixels)

            if physical_area >= area_threshold_km2:
                storm_coords = contour[:, [1, 0]].tolist()  
                pixel_count = int(np.sum(storm_pixels))
                storms_in_frame.append({
                    "mask": inside.astype(int), 
                    "contour": storm_coords,
                    "physical_area_km2": float(physical_area),
                    "pixel_count": pixel_count
                })

        frame_result = {
            "time_step": t,
            "storm_count": len(storms_in_frame),
            "storm_coordinates": [s["contour"] for s in storms_in_frame],
            "storm_masks": [s["mask"] for s in storms_in_frame],
            "storm_areas_km2": [s["physical_area_km2"] for s in storms_in_frame],
            "storm_pixel_counts": [s["pixel_count"] for s in storms_in_frame],
            "storm_durations_frames": [1.0 for _ in storms_in_frame]  
        }

        results.append(frame_result)

    return results

def calculate_storm_durations(storm_results, overlap_threshold=0.2):
    """
    Calculate storm durations by tracking storms across time steps.
    
    Parameters
    ----------
    storm_results : list
        List of dicts from detect_storms().
    overlap_threshold : float, optional
        Minimum overlap ratio to consider storms as the same (default: 0.2).
    
    Returns
    -------
    list
        List of dicts with duration information added.
    """
    if not storm_results:
        return storm_results
    
    # Track storms across time steps
    storm_tracks = {}
    next_track_id = 0
    
    for t, frame_result in enumerate(storm_results):
        current_storms = []
        for i, (contour, area) in enumerate(zip(frame_result["storm_coordinates"], frame_result["storm_areas_km2"])):
            current_storms.append({
                "contour": contour,
                "area": area,
                "storm_idx": i
            })
        
        # Match with existing tracks
        matched_current = set()
        for track_id, track in storm_tracks.items():
            if not track:
                continue
            last_time, last_idx, last_area, last_contour = track[-1]
            
            # Check previous time step
            if last_time != t - 1:
                continue
                
            # Find best match
            best_match = None
            best_overlap = 0
            
            for current_storm in current_storms:
                if current_storm["storm_idx"] in matched_current:
                    continue
                    
                # Calculate overlap
                overlap = calculate_contour_overlap(last_contour, current_storm["contour"])
                
                if overlap > best_overlap and overlap > overlap_threshold:
                    best_overlap = overlap
                    best_match = current_storm
            
            if best_match is not None:
                # Extend track
                track.append((t, best_match["storm_idx"], best_match["area"], best_match["contour"]))
                matched_current.add(best_match["storm_idx"])
        
        # Create new tracks
        for current_storm in current_storms:
            if current_storm["storm_idx"] not in matched_current:
                storm_tracks[next_track_id] = [(t, current_storm["storm_idx"], current_storm["area"], current_storm["contour"])]
                next_track_id += 1
    
    # Update durations
    for track_id, track in storm_tracks.items():
        duration = len(track)
        for time_step, storm_idx, area, contour in track:
            if storm_idx < len(storm_results[time_step]["storm_durations_frames"]):
                storm_results[time_step]["storm_durations_frames"][storm_idx] = float(duration)
    
    return storm_results

def calculate_contour_overlap(contour1, contour2, data_shape=None):
    """
    Calculate overlap between two storm contours using Jaccard index.
    
    Jaccard index is the ratio of the intersection to the union of two sets.
    
    Parameters
    ----------
    contour1 : list
        List of (x, y) coordinates for first contour.
    contour2 : list
        List of (x, y) coordinates for second contour.
    data_shape : tuple, optional
        (height, width) for creating masks. If None, will estimate from contours (default: None).
    
    Returns
    -------
    float
        Jaccard index (0 to 1).
    """
    try:
        # Convert to arrays
        c1 = np.array(contour1)
        c2 = np.array(contour2)
        
        if data_shape is None:
            max_x = max(np.max(c1[:, 0]) if c1.size > 0 else 0, 
                       np.max(c2[:, 0]) if c2.size > 0 else 0)
            max_y = max(np.max(c1[:, 1]) if c1.size > 0 else 0, 
                       np.max(c2[:, 1]) if c2.size > 0 else 0)
            data_shape = (int(max_y) + 1, int(max_x) + 1)
        
        # Create masks
        def contour_to_mask(contour, shape):
            if len(contour) == 0:
                return np.zeros(shape, dtype=bool)
            
            from matplotlib.path import Path as MplPath
            if len(shape) == 1:
                shape = (shape[0], 1)
            elif len(shape) == 0:
                shape = (1, 1)
            xg, yg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            coords = np.vstack((xg.ravel(), yg.ravel())).T
            path = MplPath(np.array(contour))
            mask = path.contains_points(coords).reshape(shape)
            return mask
        
        mask1 = contour_to_mask(c1, data_shape)
        mask2 = contour_to_mask(c2, data_shape)
        
        # Calculate Jaccard index
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        return float(jaccard)
            
    except Exception as e:
        print(f"Warning: Error calculating contour overlap: {e}")
        return 0.0

def detect_new_storm_formations(data, reflectivity_threshold=45, area_threshold_km2=10.0, dilation_iterations=5, overlap_threshold=0.2, storm_tracking_overlap_threshold=0.2, use_displacement_prediction=True, patch_size=32, patch_stride=16, patch_thresh=35.0, patch_frac=0.015, maxv=85.0, max_displacement=100, min_correlation_quality=0.5, show_progress=True):
    """
    Detect new storm formations using displacement-based prediction or simple overlap tracking.
    
    This function identifies which storms are "new" (not continuations of previous storms).
    
    Displacement-based method (use_displacement_prediction=True):
        Compute displacement vectors and fields using patch-based cross-correlation.
        Then for each time step: 
            - Predict where previous storms should be based on displacement field
            - Compare current storms with predicted positions
            - Storms that don't overlap with predictions are considered "new"
    
    Simple overlap method (use_displacement_prediction=False):
        Compare storms directly between consecutive frames
        Storms that don't overlap with previous storms are considered "new"
    
    Parameters
    ----------
    data : np.ndarray
        Radar reflectivity data of shape (T, H, W) over time.
    reflectivity_threshold : float, optional
        dBZ threshold for storm detection (default: 45).
    area_threshold_km2 : float, optional
        Minimum storm area in km² (default: 10.0).
    dilation_iterations : int, optional
        Number of dilation iterations for storm smoothing (default: 5).
    overlap_threshold : float, optional
        Overlap threshold for determining if a storm is new (default: 0.2).
    storm_tracking_overlap_threshold : float, optional
        Overlap threshold for tracking storms across time steps (default: 0.2).
    use_displacement_prediction : bool, optional
        Whether to use displacement-based prediction (default: True).
    patch_size : int, optional
        Size of patches for cross-correlation (default: 32).
    patch_stride : int, optional
        Stride between patches (default: 16).
    patch_thresh : float, optional
        Threshold for patch selection in dBZ (default: 35.0).
    patch_frac : float, optional
        Minimum fraction of pixels above threshold (default: 0.015).
    maxv : float, optional
        Maximum value for normalization (default: 85.0).
    max_displacement : int, optional
        Maximum expected displacement in pixels (default: 100).
    min_correlation_quality : float, optional
        Minimum correlation quality threshold (default: 0.5).
    
    Returns
    -------
    tuple or list
        If use_displacement_prediction=True: 
            (new_storms_summary, displacement_fields, selected_patch_centers, quality_scores)
        If use_displacement_prediction=False: 
            new_storms_summary
    """
    storm_results = detect_storms(data, reflectivity_threshold, area_threshold_km2, dilation_iterations)
    
    # Track storms across time steps
    storm_results = calculate_storm_durations(storm_results, overlap_threshold=storm_tracking_overlap_threshold)
    
    new_storms_summary = []

    if use_displacement_prediction and len(data) > 1:
        T = len(data)
        displacement_vectors = np.zeros((T-1, 2))
        quality_scores = np.zeros(T-1)
        selected_patch_centers = []
        displacement_fields = [None] * (T-1)
        
        previous_masks = []
        
        if show_progress:
            t_range = tqdm(enumerate(storm_results), desc="Processing storm formations with displacement", total=len(storm_results))
        else:
            t_range = enumerate(storm_results)
        
        for t, frame_result in t_range:
            new_count = 0
            new_coords = []
            current_masks = frame_result['storm_masks']
            current_coords = frame_result['storm_coordinates']
            current_areas = frame_result.get('storm_areas_km2', [])
            current_pixels = frame_result.get('storm_pixel_counts', [])
            current_durations = frame_result.get('storm_durations_frames', [])

            if t > 0 and len(previous_masks) > 0:
                # Predict where previous storms should be based on displacement field
                displacement_field = compute_single_displacement_field(data, t-1, 
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    max_displacement=max_displacement,
                    patch_thresh=patch_thresh,
                    patch_frac=patch_frac,
                    maxv=maxv,
                    min_correlation_quality=min_correlation_quality,
                    displacement_vectors=displacement_vectors,
                    quality_scores=quality_scores,
                    selected_patch_centers=selected_patch_centers
                )
                displacement_fields[t-1] = displacement_field
                predicted_masks = predict_storm_positions(previous_masks, displacement_field, data.shape[1:])
                
                # Check current storms against predictions
                for i, current in enumerate(current_masks):
                    is_new = True
                    for predicted in predicted_masks:
                        # Use Jaccard index for overlap
                        intersection = np.sum(current & predicted)
                        union = np.sum(current | predicted)
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        if jaccard > overlap_threshold:
                            is_new = False
                            break
                    if is_new:
                        new_count += 1
                        new_coords.append(current_coords[i])
            else:
                # All storms are new on the first frame
                new_count = len(current_masks)
                new_coords = current_coords

            new_areas = []
            new_pixels = []
            new_durations = []
            for i, coord in enumerate(new_coords):
                # Match by index
                if i < len(current_areas) and i < len(current_pixels) and i < len(current_durations):
                    new_areas.append(current_areas[i])
                    new_pixels.append(current_pixels[i])
                    new_durations.append(current_durations[i])
                else:
                    print(f"WARNING: Could not find area/pixels/duration for coordinate {coord} at t={t}")
                    new_areas.append(None)
                    new_pixels.append(None)
                    new_durations.append(None)
            
            new_storms_summary.append({
                'time_step': t,
                'new_storm_count': new_count,
                'new_storm_coordinates': new_coords,
                'storm_areas_km2': new_areas,
                'storm_pixel_counts': new_pixels,
                'storm_durations_frames': new_durations
            })
            
            # Update for next iteration
            previous_masks = current_masks
        
        return new_storms_summary, displacement_fields, selected_patch_centers, quality_scores
    else:
        # Simple overlap-based method without displacement prediction
        previous_masks = []
        
        for t, frame_result in enumerate(storm_results):
            new_count = 0
            new_coords = []
            current_masks = frame_result['storm_masks']
            current_coords = frame_result['storm_coordinates']
            current_areas = frame_result.get('storm_areas_km2', [])
            current_pixels = frame_result.get('storm_pixel_counts', [])
            current_durations = frame_result.get('storm_durations_frames', [])
            
            if t > 0 and len(previous_masks) > 0:
                # Check each current storm against previous storms
                for i, current in enumerate(current_masks):
                    is_new = True
                        for previous in previous_masks:
                        # Use Jaccard index for overlap
                        intersection = np.sum(current & previous)
                        union = np.sum(current | previous)
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        if jaccard > overlap_threshold:
                                is_new = False
                                break
                    if is_new:
                        new_count += 1
                        new_coords.append(current_coords[i])
            else:
                # All storms are new on the first frame
                new_count = len(current_masks)
                new_coords = current_coords
            
            new_areas = []
            new_pixels = []
            new_durations = []
            for i, coord in enumerate(new_coords):
                # Match by index
                if i < len(current_areas) and i < len(current_pixels) and i < len(current_durations):
                    new_areas.append(current_areas[i])
                    new_pixels.append(current_pixels[i])
                    new_durations.append(current_durations[i])
                else:
                    print(f"WARNING: Could not find area/pixels/duration for coordinate {coord} at t={t}")
                    new_areas.append(None)
                    new_pixels.append(None)
                    new_durations.append(None)
            
            new_storms_summary.append({
                'time_step': t,
                'new_storm_count': new_count,
                'new_storm_coordinates': new_coords,
                'storm_areas_km2': new_areas,
                'storm_pixel_counts': new_pixels,
                'storm_durations_frames': new_durations
            })
            
            previous_masks = current_masks
        
        return new_storms_summary

def compute_single_displacement_field(data, t, patch_size=32, patch_stride=16, max_displacement=100, 
                                    patch_thresh=35.0, patch_frac=0.015, maxv=85.0, min_correlation_quality=0.50,
                                    displacement_vectors=None, quality_scores=None, selected_patch_centers=None):
    """
    Compute displacement vectors and fields using patch-based cross-correlation for a single time step.
    
    This function calculates how radar echoes move between consecutive frames.
    
    Patch displacement calculation:
       - Divides frames into overlapping patches
       - Finds displacement using cross-correlation
       - Uses patches with sufficient reflectivity
    
    Temporal smoothing:
       - Uses moving average to prevent sudden changes
    
    Displacement field creation:
       - Interpolates sparse displacements into dense fields
    
    Parameters
    ----------
    data : np.ndarray
        Radar reflectivity data of shape (T, H, W).
    t : int
        Time step index (0 to T-2).
    patch_size : int, optional
        Size of patches for cross-correlation in pixels (default: 32).
    patch_stride : int, optional
        Stride between patch centers in pixels (default: 16).
    max_displacement : int, optional
        Maximum allowed displacement magnitude in pixels (default: 100).
    patch_thresh : float, optional
        Minimum reflectivity threshold for patch selection in dBZ (default: 35.0).
    patch_frac : float, optional
        Minimum fraction of patch pixels above threshold (default: 0.015).
    maxv : float, optional
        Maximum reflectivity value for normalization (default: 85.0).
    min_correlation_quality : float, optional
        Minimum correlation quality threshold (0-1, default: 0.5).
    
    Returns
    -------
    np.ndarray
        Displacement field of shape (H, W, 2) for time step t.
    """
    T, H, W = data.shape
    
    if t >= T - 1:
        return np.zeros((H, W, 2))
    
    n_patches_y = max(1, (H - 1) // patch_stride + 1)
    n_patches_x = max(1, (W - 1) // patch_stride + 1)
    
    patch_thresh_normalized = patch_thresh / (maxv + 1e-6)
    
    # Quality filtering helper function
    def compute_correlation_quality(correlation, peak_y, peak_x):
        """Compute correlation quality as ratio of peak to maximum possible correlation."""
        max_correlation = correlation.max()
        if max_correlation > 0:
            return correlation[peak_y, peak_x] / max_correlation
        return 0.0

        frame1 = data[t]
        frame2 = data[t+1]
        
        patch_displacements = []
        patch_positions = []
        
        for i in range(n_patches_y):
            for j in range(n_patches_x):
            # Get patch positions
                y_start = i * patch_stride
                x_start = j * patch_stride
                y_end = y_start + patch_size
                x_end = x_start + patch_size
                
            # Get patch center
                patch_center_y = y_start + patch_size // 2
                patch_center_x = x_start + patch_size // 2
                
            # Extract patches with padding
                y_start_actual = max(0, y_start)
                x_start_actual = max(0, x_start)
                y_end_actual = min(H, y_end)
                x_end_actual = min(W, x_end)
                
            # Extract data within bounds
                patch1_data = frame1[y_start_actual:y_end_actual, x_start_actual:x_end_actual]
                patch2_data = frame2[y_start_actual:y_end_actual, x_start_actual:x_end_actual]
                
            # Create full size patches
                patch1 = np.zeros((patch_size, patch_size))
                patch2 = np.zeros((patch_size, patch_size))
                
            # Get padding offsets
                y_pad_start = max(0, -y_start)
                x_pad_start = max(0, -x_start)
                y_pad_end = patch_size - max(0, y_end - H)
                x_pad_end = patch_size - max(0, x_end - W)
                
            # Fill patches with data
                patch1[y_pad_start:y_pad_end, x_pad_start:x_pad_end] = patch1_data
                patch2[y_pad_start:y_pad_end, x_pad_start:x_pad_end] = patch2_data
                
            # Skip low variation patches
                if np.std(patch1) < 1e-6 or np.std(patch2) < 1e-6:
                    continue
                
            # High reflectivity patch selection
                patch_normalized = np.maximum(patch1, 0) / (maxv + 1e-6)
                total_pix = patch_normalized.size
                n_above = (patch_normalized > patch_thresh_normalized).sum()
                if n_above / total_pix < patch_frac:
                    continue
                
                # Normalize patches for better correlation
                patch1_norm = (patch1 - np.mean(patch1)) / (np.std(patch1) + 1e-8)
                patch2_norm = (patch2 - np.mean(patch2)) / (np.std(patch2) + 1e-8)
                
            # Correlation computation
                correlation = correlate2d(patch1_norm, patch2_norm, mode='full')
                
                # Find the peak of correlation
                center_y, center_x = correlation.shape[0] // 2, correlation.shape[1] // 2
                
                # Search within max_displacement range
                y_start_search = max(0, center_y - max_displacement)
                y_end_search = min(correlation.shape[0], center_y + max_displacement + 1)
                x_start_search = max(0, center_x - max_displacement)
                x_end_search = min(correlation.shape[1], center_x + max_displacement + 1)
                
                search_region = correlation[y_start_search:y_end_search, x_start_search:x_end_search]
                
                if search_region.size == 0:
                    continue
                    
                max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
                peak_y = y_start_search + max_idx[0]
                peak_x = x_start_search + max_idx[1]
                
                # Calculate displacement to the new position for this patch
            u = -(peak_x - center_x)  # horizontal displacement
            v = -(peak_y - center_y)  # vertical displacement
                
            # Check displacement magnitude and reject if too large
                displacement_magnitude = np.sqrt(u**2 + v**2)
                if displacement_magnitude > max_displacement:
                    continue
                
            # Only use patches with good correlations
                quality = compute_correlation_quality(correlation, peak_y, peak_x)
                if quality < min_correlation_quality:
                    continue
                
                if np.isnan(u) or np.isinf(u) or np.isnan(v) or np.isinf(v):
                    continue
                
                patch_displacements.append([u, v])
                patch_positions.append([patch_center_y, patch_center_x])
        
    # Calculate global displacement vector and apply temporal smoothing 
        if len(patch_displacements) > 0:
            patch_displacements = np.array(patch_displacements)
            current_displacement = np.mean(patch_displacements, axis=0)
            
        # Apply temporal smoothing
        if displacement_vectors is not None and t > 0:
            # Weighted average with previous displacement
            alpha = 0.7
                displacement_vectors[t] = alpha * current_displacement + (1 - alpha) * displacement_vectors[t-1]
            else:
                displacement_vectors[t] = current_displacement
            
        # Compute quality score based on displacement consistency 
            displacement_std = np.std(patch_displacements, axis=0)
            displacement_magnitude = np.linalg.norm(displacement_vectors[t])
            quality_scores[t] = 1.0 / (1.0 + np.mean(displacement_std) / (displacement_magnitude + 1e-6))
        
        # Store patch centers
        if selected_patch_centers is not None:
            selected_patch_centers.append(patch_positions)
        
        # Create displacement field
        displacement_field = create_displacement_field(patch_displacements, patch_positions, (H, W), None, max_displacement)
    else:
        # Handle no patches case 
        if displacement_vectors is not None and t > 0:
                displacement_vectors[t] = displacement_vectors[t-1]
                quality_scores[t] = 0.5
            else:
            displacement_vectors[t] = np.array([0.0, 0.0])
                quality_scores[t] = 0.0
        
        # Store empty patch centers
        if selected_patch_centers is not None:
            selected_patch_centers.append([])
        
        # Create uniform displacement field using temporally smoothed vector
        displacement_field = np.full((H, W, 2), displacement_vectors[t])
    
    return displacement_field

def create_displacement_field(patch_displacements, patch_positions, field_shape, previous_displacement_field=None, max_displacement=100):
    """
    Create a dense displacement field by interpolating sparse patch displacement vectors.
    
    This function takes displacement vectors from patch centers and creates a displacement
    field for every pixel in the image using distance-weighted interpolation.
    
    Interpolation process:
        For each pixel in the image:
            - Find all patch centers within max_interpolation_distance
            - Calculate weights based on inverse distance (closer patches have higher weight)
            - Compute weighted average of nearby patch displacements
        If no nearby patches exist:
            - Use previous displacement field as fallback (if reasonable magnitude)
            - Otherwise use zero displacement
    
    Smoothing:
        Applies Gaussian smoothing to reduce interpolation noise.
    
    Parameters
    ----------
    patch_displacements : list
        List of [u, v] displacement vectors for each patch that passed quality filtering.
    patch_positions : list
        List of [y, x] coordinates of patch centers.
    field_shape : tuple
        (H, W) shape of the output displacement field.
    previous_displacement_field : np.ndarray, optional
        Displacement field from previous time step for fallback (default: None).
    max_displacement : int, optional
        Maximum displacement magnitude for fallback validation (default: 100).
    
    Returns
    -------
    np.ndarray
        Dense displacement field of shape (H, W, 2) where:
        - displacement_field[y, x, 0] = horizontal displacement (u component)
        - displacement_field[y, x, 1] = vertical displacement (v component)
    """
    H, W = field_shape
    displacement_field = np.zeros((H, W, 2))
    
    if len(patch_displacements) == 0:
        return displacement_field
    
    patch_displacements = np.array(patch_displacements)
    patch_positions = np.array(patch_positions)
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    

    for comp in range(2):
        field_comp = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):

                    distances = np.sqrt((patch_positions[:, 0] - i)**2 + (patch_positions[:, 1] - j)**2)
                

                    min_dist = np.min(distances)
                    if min_dist < 1e-6:

                        nearest_idx = np.argmin(distances)
                        field_comp[i, j] = patch_displacements[nearest_idx, comp]
                    else:
                        max_interpolation_distance = 30
                        nearby_mask = distances <= max_interpolation_distance
                        
                        if np.any(nearby_mask):

                            nearby_distances = distances[nearby_mask]
                            nearby_displacements = patch_displacements[nearby_mask, comp]
                            weights = 1.0 / (nearby_distances + 1e-6)
                            field_comp[i, j] = np.sum(nearby_displacements * weights) / np.sum(weights)
                        else:

                            if previous_displacement_field is not None:
                                prev_disp = previous_displacement_field[i, j, comp]

                                if abs(prev_disp) < max_displacement:
                                    field_comp[i, j] = prev_disp
                                else:
                                    field_comp[i, j] = 0.0
                            else:

                                field_comp[i, j] = 0.0
        

        from scipy.ndimage import gaussian_filter
        field_comp_smooth = gaussian_filter(field_comp, sigma=0.5)
        displacement_field[:, :, comp] = field_comp_smooth
    
    return displacement_field

def predict_storm_positions(previous_storms, displacement_field, data_shape):
    """
    Predict where storms from the previous frame should be in the current frame.
    
    This function uses the displacement field to predict where each storm
    pixel should move in the next frame. Each pixel in a storm is moved according to
    its local displacement vector.
    
    Prediction process:
        For each storm from the previous frame:
            - Get storm pixel coordinates
            - Look up displacement vectors
            - Move pixels to new positions
            - Create new storm mask
            - Apply dilation
    
    Parameters
    ----------
    previous_storms : list
        List of binary storm masks from previous frame (each mask is H×W boolean array).
    displacement_field : np.ndarray
        Dense displacement field of shape (H, W, 2) with (u, v) components at each pixel.
    data_shape : tuple
        (H, W) shape of the radar data.
    
    Returns
    -------
    list
        List of predicted storm masks showing where storms should be
        in the current frame based on wind displacement.
    """
    H, W = data_shape
    
    predicted_masks = []
    
    for storm_mask in previous_storms:

        y_coords, x_coords = np.where(storm_mask)
        
        if len(y_coords) == 0:
            continue
            
        storm_displacement_u = displacement_field[y_coords, x_coords, 0]
        storm_displacement_v = displacement_field[y_coords, x_coords, 1]
        

        new_y = y_coords + storm_displacement_v
        new_x = x_coords + storm_displacement_u
        
        if np.any(np.isnan(new_y)) or np.any(np.isnan(new_x)) or np.any(np.isinf(new_y)) or np.any(np.isinf(new_x)):
            continue

        new_y = np.clip(new_y, 0, H-1).astype(int)
        new_x = np.clip(new_x, 0, W-1).astype(int)

        predicted_mask = np.zeros((H, W), dtype=bool)
        predicted_mask[new_y, new_x] = True

        predicted_mask = binary_dilation(predicted_mask, iterations=2)
        
        predicted_masks.append(predicted_mask)
    
    return predicted_masks

def analyze_incorrect_initiation_reflectivity(incorrect_initiations_data, true_data, predicted_data, data_shape):
    """
    Analyze reflectivity values in areas where incorrect storm initiations were predicted.
    
    This function examines true radar data in areas where predicted storms
    were classified as "incorrect initiations" to understand if the predicted storms correspond to
    high reflectivity areas in the true data.

    Parameters
    ----------
    incorrect_initiations_data : list
        List of dictionaries with 'time_step', 'storm_mask', and 'storm_idx' for incorrect initiations.
    true_data : np.ndarray
        True radar reflectivity data of shape (T, H, W).
    predicted_data : np.ndarray
        Predicted radar reflectivity data of shape (T, H, W).
    data_shape : tuple
        (height, width) shape of the data.

    Returns
    -------
    dict
        Dictionary with reflectivity analysis results:
        {
            "true_reflectivity_above_40": int,   # Very high reflectivity (>40 dBZ) in the true data
            "true_reflectivity_35_40": int,      # High reflectivity (35-40 dBZ) in the true data
            "true_reflectivity_30_35": int,      # Medium-high reflectivity (30-35 dBZ) in the true data
            "true_reflectivity_below_30": int,   # Low reflectivity (<30 dBZ) in the true data
            "avg_predicted_reflectivity_above_45": float, # Average reflectivity of pixels ≥45 dBZ in the predicted data
            "avg_actual_reflectivity_corresponding": float, # Average reflectivity of pixels in corresponding locations in the true data
            "reflectivity_difference": float # Difference (predicted - actual)
        }
    """
    reflectivity_ranges = {
        "true_reflectivity_above_40": 0,
        "true_reflectivity_35_40": 0,
        "true_reflectivity_30_35": 0,
        "true_reflectivity_below_30": 0
    }
    
    total_analyzed = len(incorrect_initiations_data)
    total_with_valid_reflectivity = 0
    
    predicted_high_reflectivities = []
    true_corresponding_reflectivities = []
    
    for storm_data in incorrect_initiations_data:
        time_step = storm_data['time_step']
        storm_mask = storm_data['storm_mask']
        
        if time_step >= true_data.shape[0]:
            continue
            
        # Get the true and predicted radar data for this time step
        true_frame = true_data[time_step]
        predicted_frame = predicted_data[time_step]
        
        if storm_mask.shape != true_frame.shape or storm_mask.shape != predicted_frame.shape:
            print(f"Warning: Mask shape {storm_mask.shape} doesn't match data shape {true_frame.shape} or {predicted_frame.shape}")
            continue
        
        # Extract high reflectivity pixels from predicted data
        high_reflectivity_mask = (predicted_frame >= 45) & storm_mask
        predicted_high_reflectivity = predicted_frame[high_reflectivity_mask]
        
        # Extract corresponding pixels in true data
        true_corresponding_reflectivity = true_frame[high_reflectivity_mask]
        
        if len(predicted_high_reflectivity) == 0:
            continue
            
        total_with_valid_reflectivity += 1
        
        # Mean reflectivity of high reflectivity pixels 
        mean_predicted_high_reflectivity = np.mean(predicted_high_reflectivity)
        mean_true_corresponding_reflectivity = np.mean(true_corresponding_reflectivity)
        predicted_high_reflectivities.append(mean_predicted_high_reflectivity)
        true_corresponding_reflectivities.append(mean_true_corresponding_reflectivity)
        
        # Categorize based on mean true reflectivity in corresponding locations
        if mean_true_corresponding_reflectivity > 40:
            reflectivity_ranges["true_reflectivity_above_40"] += 1
        elif mean_true_corresponding_reflectivity >= 35:
            reflectivity_ranges["true_reflectivity_35_40"] += 1
        elif mean_true_corresponding_reflectivity >= 30:
            reflectivity_ranges["true_reflectivity_30_35"] += 1
        else:
            reflectivity_ranges["true_reflectivity_below_30"] += 1
    
    # Calculate average reflectivities 
    avg_predicted_high_reflectivity = np.mean(predicted_high_reflectivities) if predicted_high_reflectivities else 0.0
    avg_true_corresponding_reflectivity = np.mean(true_corresponding_reflectivities) if true_corresponding_reflectivities else 0.0
    reflectivity_difference = avg_predicted_high_reflectivity - avg_true_corresponding_reflectivity
    
    return {
        "incorrect_initiation_reflectivity_analysis": {
            "description": "Analysis of high-reflectivity pixels in predicted storm areas and corresponding true data locations.",
            **reflectivity_ranges,
            "avg_predicted_reflectivity_above_45": float(avg_predicted_high_reflectivity),
            "avg_actual_reflectivity_corresponding": float(avg_true_corresponding_reflectivity),
            "reflectivity_difference": float(reflectivity_difference)
        }
    }

def evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold=0.2, true_data_shape=None, true_data=None, predicted_data=None):
    """
    Compares predicted and true new storm initiations.

    Parameters
    ----------
    new_storms_pred : list
        List of dicts from detect_new_storm_formations(pred_composite_reflectivity).
    new_storms_true : list
        List of dicts from detect_new_storm_formations(true_composite_reflectivity).
    overlap_threshold : float, optional
        Minimum overlap ratio to count as a match (default: 0.2).

    Returns
    -------
    dict
        Dictionary with counts and statistics:
        {
            "correct": int,  # matched at same time
            "early": int,    # matched one time step before
            "late": int,     # matched one time step after
            "incorrect_initiations": int,  # predicted storm initiations with no match in ±1 time step
            "total_true": int,  # total true new storms
            "total_pred": int,  # total predicted new storms
            "correct_over_true": float, # ratio of correct to total true new storms
            "correct_over_pred": float, # ratio of correct to total predicted new storms
            "anytime_ratio": float, # ratio of (correct+early+late) to total true new storms
            "incorrect_initiation_ratio": float, # ratio of incorrect initiations to total predicted new storms
            "statistics": dict,  # statistics for each category
        }
    """
    # Helper to get mask for a storm contour
    def contour_to_mask(contour, shape):
        """
        Convert contour coordinates to binary mask.
        
        Parameters:
        - contour: list, list of (x, y) coordinates
        - shape: tuple, (height, width) of the output mask
        
        Returns:
        - np.ndarray: binary mask of the contour
        """
        from matplotlib.path import Path as MplPath
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif len(shape) == 0:
            shape = (1, 1)
        xg, yg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        path = MplPath(np.array(contour))
        mask = path.contains_points(coords).reshape(shape)
        return mask
    
    
    def build_storm_masks(new_storms, data_shape):
        """
        Build lookup table for storm masks and metadata.
        
        Parameters:
        - new_storms: list, list of storm dictionaries
        - data_shape: tuple, (height, width) for mask creation
        
        Returns:
        - dict: lookup table mapping time_step to list of storm masks with metadata
        """
        lookup = {}
        for entry in new_storms:
            t = entry["time_step"]
            masks = []
            for i, contour in enumerate(entry["new_storm_coordinates"]):
                mask = contour_to_mask(contour, data_shape)
                area = entry.get("storm_areas_km2", [15.0])[i] if i < len(entry.get("storm_areas_km2", [])) else 15.0
                pixel_count = entry.get("storm_pixel_counts", [100])[i] if i < len(entry.get("storm_pixel_counts", [])) else 100
                duration = entry.get("storm_durations_frames", [1.0])[i] if i < len(entry.get("storm_durations_frames", [])) else 1.0
                masks.append({"mask": mask, "used": False, "area": area, "pixel_count": pixel_count, "duration": duration})
            lookup[t] = masks
        return lookup

    if true_data_shape is not None:
        data_shape = true_data_shape
                else:
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
                        x, y = arr[0], arr[1]
                        max_x = max(max_x, int(x))
                        max_y = max(max_y, int(y))
                    elif arr.ndim == 2 and arr.shape[1] == 2:
                        max_x = max(max_x, int(arr[:, 0].max()))
                        max_y = max(max_y, int(arr[:, 1].max()))
        if not found:
            default_stats = {
                "correct": {"avg_area_km2": None, "avg_duration_frames": None, "avg_pixels": None, "storm_count": 0},
                "early": {"avg_area_km2": None, "avg_duration_frames": None, "avg_pixels": None, "storm_count": 0},
                "late": {"avg_area_km2": None, "avg_duration_frames": None, "avg_pixels": None, "storm_count": 0},
                "incorrect_initiations": {"avg_area_km2": None, "avg_duration_frames": None, "avg_pixels": None, "storm_count": 0},
                "true_storms": {"avg_area_km2": None, "avg_duration_frames": None, "avg_pixels": None, "storm_count": 0}
            }
            return {
                "correct": 0, 
                "early": 0, 
                "late": 0, 
                "incorrect_initiations": sum(e["new_storm_count"] for e in new_storms_pred), 
                "total_true": 0, 
                "total_pred": sum(e["new_storm_count"] for e in new_storms_pred), 
                "correct_over_true": 0.0, 
                "correct_over_pred": 0.0, 
                "anytime_ratio": 0.0, 
                "incorrect_initiation_ratio": 0.0,
                "statistics": default_stats
            }
        data_shape = (max_y + 1, max_x + 1)


    pred_lookup = build_storm_masks(new_storms_pred, data_shape)
    true_lookup = build_storm_masks(new_storms_true, data_shape)

    correct = 0
    early = 0
    late = 0
    matched_pred = set() 
    matched_true = set() 
    
    correct_areas = []
    correct_durations = []
    correct_pixels = []
    early_areas = []
    early_durations = []
    early_pixels = []
    late_areas = []
    late_durations = []
    late_pixels = []
    incorrect_initiation_areas = []
    incorrect_initiation_durations = []
    incorrect_initiation_pixels = []
    true_storm_areas = []
    true_storm_durations = []
    true_storm_pixels = []

    for t, true_storms in true_lookup.items():
        for i, true_storm in enumerate(true_storms):
 
            true_storm_areas.append(true_storm["area"])
            true_storm_durations.append(true_storm["duration"])
            true_storm_pixels.append(true_storm["pixel_count"])
            
            found = False
            for dt, label in zip([0, -1, 1], ["correct", "early", "late"]):
                tt = t + dt
                if tt in pred_lookup:
                    for j, pred_storm in enumerate(pred_lookup[tt]):
                        if (tt, j) in matched_pred:
                continue
                        # Overlap using Jaccard index
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        union = np.sum(true_storm["mask"] | pred_storm["mask"])
                        if union == 0:
                continue
                        jaccard = intersection / union
                        if jaccard >= overlap_threshold:
                            pred_area = pred_storm["area"]
                            pred_duration = pred_storm["duration"]
                            
                            if label == "correct":
                                correct += 1
                                correct_areas.append(pred_area)
                                correct_durations.append(pred_duration)
                                correct_pixels.append(pred_storm["pixel_count"])
                            elif label == "early":
                                early += 1
                                early_areas.append(pred_area)
                                early_durations.append(pred_duration)
                                early_pixels.append(pred_storm["pixel_count"])
                            elif label == "late":
                                late += 1
                                late_areas.append(pred_area)
                                late_durations.append(pred_duration)
                                late_pixels.append(pred_storm["pixel_count"])
                            
                            matched_pred.add((tt, j))
                            matched_true.add((t, i))
                            found = True
                            break
                if found:
                    break

    # Incorrect initiations are predicted storms not matched within one time step
    incorrect_initiations = 0
    incorrect_initiations_data = []  
    
    for t, pred_storms in pred_lookup.items():
        for j, pred_storm in enumerate(pred_storms):
            if (t, j) in matched_pred:
                continue

            matched = False
            for dt in [-1, 0, 1]:
                tt = t + dt
                if tt in true_lookup:
                    for i, true_storm in enumerate(true_lookup[tt]):
                        if (tt, i) in matched_true:
                            continue
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        union = np.sum(true_storm["mask"] | pred_storm["mask"])
                        if union == 0:
                            continue
                        jaccard = intersection / union
                        if jaccard >= overlap_threshold:
                            matched = True
                        break
                if matched:
                    break
            if not matched:
                    incorrect_initiations += 1

                incorrect_initiation_areas.append(pred_storm["area"])
                incorrect_initiation_durations.append(pred_storm["duration"])
                incorrect_initiation_pixels.append(pred_storm["pixel_count"])
                
                incorrect_initiations_data.append({
                    'time_step': t,
                    'storm_mask': pred_storm["mask"],
                    'storm_idx': j
                })

    total_true = sum(len(v) for v in true_lookup.values())
    total_pred = sum(len(v) for v in pred_lookup.values())

    correct_over_true = correct / total_true if total_true > 0 else 0.0
    correct_over_pred = correct / total_pred if total_pred > 0 else 0.0
    anytime_ratio = (correct + early + late) / total_true if total_true > 0 else 0.0
    incorrect_initiation_ratio = incorrect_initiations / total_pred if total_pred > 0 else 0.0


    def safe_mean(values):
        """
        Calculate mean of values, returning None if no valid values.
        
        Parameters
        ----------
        values : list
            List of numeric values.
        
        Returns
        -------
        float or None
            Mean of values, or None if no valid values.
        """
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None
    
    statistics = {
        "correct": {
            "avg_area_km2": safe_mean(correct_areas),
            "avg_duration_frames": safe_mean(correct_durations),
            "avg_pixels": safe_mean(correct_pixels),
            "storm_count": len(correct_areas)
        },
        "early": {
            "avg_area_km2": safe_mean(early_areas),
            "avg_duration_frames": safe_mean(early_durations),
            "avg_pixels": safe_mean(early_pixels),
            "storm_count": len(early_areas)
        },
        "late": {
            "avg_area_km2": safe_mean(late_areas),
            "avg_duration_frames": safe_mean(late_durations),
            "avg_pixels": safe_mean(late_pixels),
            "storm_count": len(late_areas)
        },
        "incorrect_initiations": {
            "avg_area_km2": safe_mean(incorrect_initiation_areas),
            "avg_duration_frames": safe_mean(incorrect_initiation_durations),
            "avg_pixels": safe_mean(incorrect_initiation_pixels),
            "storm_count": len(incorrect_initiation_areas)
        },
        "true_storms": {
            "avg_area_km2": safe_mean(true_storm_areas),
            "avg_duration_frames": safe_mean(true_storm_durations),
            "avg_pixels": safe_mean(true_storm_pixels),
            "storm_count": len(true_storm_areas)
        }
    }

    reflectivity_analysis = {}
    if true_data is not None and predicted_data is not None and len(incorrect_initiations_data) > 0:
        reflectivity_analysis = analyze_incorrect_initiation_reflectivity(
            incorrect_initiations_data, true_data, predicted_data, data_shape
        )
    
    result = {
        "correct": int(correct),
        "early": int(early),
        "late": int(late),
        "incorrect_initiations": int(incorrect_initiations),
        "total_true": int(total_true),
        "total_pred": int(total_pred),
        "correct_over_true": float(correct_over_true),
        "correct_over_pred": float(correct_over_pred),
        "anytime_ratio": float(anytime_ratio),
        "incorrect_initiation_ratio": float(incorrect_initiation_ratio),
        "statistics": statistics
    }
    
    if reflectivity_analysis:
        result.update(reflectivity_analysis)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Evaluate new storm predictions from .npy files and save results.
        This script loads prediction and target arrays (N, C, H, W),
        reduces them to (T, H, W) by taking the max over the channel dimension (composite reflectivity),
        then runs detect_new_storm_formations and evaluate_new_storm_predictions.
        Results are printed and saved as JSON.
        """
    )
    parser.add_argument('--preds', type=str, required=True, help='Path to predicted storms .npy file (shape: N, C, H, W or N, H, W)')
    parser.add_argument('--targets', type=str, required=True, help='Path to true storms .npy file (shape: N, C, H, W or N, H, W)')
    parser.add_argument('--out', type=str, required=True, help='Output JSON file for evaluation results')
    parser.add_argument('--overlap_threshold', type=float, default=0.2, help='Overlap threshold for matching predicted and true storms (default: 0.2)')
    parser.add_argument('--storm_tracking_overlap_threshold', type=float, default=0.2, help='Overlap threshold for tracking storms across time steps (default: 0.2)')
    parser.add_argument('--reflectivity_threshold', type=float, default=45, help='Reflectivity threshold for storm detection (default: 45)')
    parser.add_argument('--area_threshold_km2', type=float, default=10.0, help='Area threshold for storm detection in km² (default: 10.0)')
    parser.add_argument('--dilation_iterations', type=int, default=5, help='Dilation iterations for storm detection (default: 5)')
   
    parser.add_argument('--use_displacement_prediction', action='store_true', default=True, help='Use displacement-based prediction for new storm detection (default: True)')
    parser.add_argument('--no_displacement_prediction', action='store_true', help='Disable displacement-based prediction (use overlap-based method)')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size for displacement computation (default: 32)')
    parser.add_argument('--patch_stride', type=int, default=16, help='Patch stride for displacement computation (default: 16)')
    parser.add_argument('--patch_thresh', type=float, default=35.0, help='Threshold for patch selection in dBZ (default: 35.0)')
    parser.add_argument('--patch_frac', type=float, default=0.015, help='Minimum fraction of pixels above threshold (default: 0.015)')
    parser.add_argument('--maxv', type=float, default=85.0, help='Maximum value for normalization (default: 85.0)')


    args = parser.parse_args()

    # Load shape and dtype from metadata files 
    def load_memmap_with_meta(array_path):
        """
        Load numpy array with metadata support.
        
        Parameters
        ----------
        array_path : str
            Path to the .npy file.
        
        Returns
        -------
        np.ndarray
            Loaded array.
        """
        meta_path = array_path.replace('.npy', '_meta.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            shape = tuple(meta['shape'])
            dtype = str(meta['dtype'])
            return np.memmap(array_path, dtype=dtype, mode='r', shape=shape)
        else:
            return np.load(array_path, mmap_mode='r')

    pred = load_memmap_with_meta(args.preds)
    tgt = load_memmap_with_meta(args.targets)


    if pred.ndim == 4:
        pred_composite = np.max(pred, axis=1)
        tgt_composite = np.max(tgt, axis=1)
    elif pred.ndim == 3:
        pred_composite = pred
        tgt_composite = tgt
    else:
        raise ValueError("Input arrays must be of shape (N, C, H, W) or (N, H, W)")


    def detect_with_progress(data, **kwargs):
        """
        Detect storms with progress tracking.
        
        Parameters
        ----------
        data : np.ndarray
            Radar data.
        **kwargs
            Additional keyword arguments passed to detect_new_storm_formations.
        
        Returns
        -------
        result
            Result from detect_new_storm_formations.
        """
        storms = []

        storm_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['reflectivity_threshold', 'area_threshold_km2', 'dilation_iterations']}
        
 
        desc = kwargs.get('desc', 'Detecting storms')
        for t in tqdm(range(data.shape[0]), desc=f"{desc} - Frame detection"):
            storms.append(detect_storms(data[t:t+1], **storm_kwargs)[0])
        
 
        use_disp = kwargs.get('use_displacement_prediction', False)
        method = "with displacement tracking" if use_disp else "with overlap tracking"
        return detect_new_storm_formations(data, **{k: v for k, v in kwargs.items() if k != 'desc'}, show_progress=True)


    use_displacement_prediction = args.use_displacement_prediction and not args.no_displacement_prediction
    

    
    if use_displacement_prediction:
        pred_result = detect_with_progress(
            pred_composite,
            reflectivity_threshold=args.reflectivity_threshold,
            area_threshold_km2=args.area_threshold_km2,
            dilation_iterations=args.dilation_iterations,
            overlap_threshold=args.overlap_threshold,
            storm_tracking_overlap_threshold=args.storm_tracking_overlap_threshold,
            use_displacement_prediction=True,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_thresh=args.patch_thresh,
            patch_frac=args.patch_frac,
            maxv=args.maxv,
            desc='Pred storms')
    else:
        pred_result = detect_with_progress(
            pred_composite,
            reflectivity_threshold=args.reflectivity_threshold,
            area_threshold_km2=args.area_threshold_km2,
            dilation_iterations=args.dilation_iterations,
            overlap_threshold=args.overlap_threshold,
            storm_tracking_overlap_threshold=args.storm_tracking_overlap_threshold,
            use_displacement_prediction=False,
            desc='Pred storms')
    
    if use_displacement_prediction:
        tgt_result = detect_with_progress(
            tgt_composite,
            reflectivity_threshold=args.reflectivity_threshold,
            area_threshold_km2=args.area_threshold_km2,
            dilation_iterations=args.dilation_iterations,
            overlap_threshold=args.overlap_threshold,
            storm_tracking_overlap_threshold=args.storm_tracking_overlap_threshold,
            use_displacement_prediction=True,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_thresh=args.patch_thresh,
            patch_frac=args.patch_frac,
            maxv=args.maxv,
            desc='True storms')
    else:
        tgt_result = detect_with_progress(
            tgt_composite,
            reflectivity_threshold=args.reflectivity_threshold,
            area_threshold_km2=args.area_threshold_km2,
            dilation_iterations=args.dilation_iterations,
            overlap_threshold=args.overlap_threshold,
            storm_tracking_overlap_threshold=args.storm_tracking_overlap_threshold,
            use_displacement_prediction=False,
            desc='True storms')


    if use_displacement_prediction and len(pred_composite) > 1:
        pred_storms, _, pred_patch_centers, pred_quality_scores = pred_result
        tgt_storms, _, tgt_patch_centers, tgt_quality_scores = tgt_result
    else:
        pred_storms = pred_result
        tgt_storms = tgt_result


    storm_results = evaluate_new_storm_predictions(pred_storms, tgt_storms, overlap_threshold=args.overlap_threshold, true_data_shape=tgt_composite.shape[1:], true_data=tgt_composite, predicted_data=pred_composite)
    

    forecasting_metrics = compute_forecasting_metrics(pred_composite, tgt_composite, maxv=args.maxv, eps=1e-6)
    

    results = {
        "storm_initiation_metrics": storm_results,
        "forecasting_metrics": forecasting_metrics
    }
    
    print("\n Storm Initiation Metrics ")
    print(json.dumps(storm_results, indent=2))

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)