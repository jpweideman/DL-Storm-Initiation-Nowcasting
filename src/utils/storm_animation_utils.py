import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from matplotlib.path import Path
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from src.utils.storm_utils import detect_storms, compute_polar_pixel_areas, detect_new_storm_formations, compute_displacement_vectors, predict_storm_positions

def animate_storms(data, reflectivity_threshold=45, area_threshold_km2=10.0, dilation_iterations=5, interval=100):
    """
    Animate storm detection over time from radar reflectivity data using physical area calculations.
    
    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - reflectivity_threshold: dBZ threshold for storm detection (default: 45)
    - area_threshold_km2: minimum storm area in km² (default: 10.0)
    - dilation_iterations: dilation iterations for storm smoothing (default: 5)
    - interval: animation interval in milliseconds (default: 100)
    """
    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap("jet")
    img = ax.imshow(data[0], cmap=cmap, vmin=0, vmax=80)
    plt.colorbar(img, ax=ax, label="Reflectivity (dBZ)")
    title = ax.set_title("Radar Reflectivity - Time Step 0")
    storm_lines = []
    # Compute pixel areas for the radar geometry
    pixel_areas = compute_polar_pixel_areas(data.shape[1:])
    
    def update(frame):
        nonlocal storm_lines
        frame_data = data[frame]
        img.set_data(frame_data)
        title.set_text(f"Radar Reflectivity - Time Step {frame}")
        for line in storm_lines:
            line.remove()
        storm_lines = []
        mask = frame_data > reflectivity_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
        contours = find_contours(dilated_mask.astype(float), 0.5)
        for contour in contours:
            path = Path(contour[:, ::-1])
            xg, yg = np.meshgrid(np.arange(frame_data.shape[1]), np.arange(frame_data.shape[0]))
            coords = np.vstack((xg.ravel(), yg.ravel())).T
            inside = path.contains_points(coords).reshape(frame_data.shape)
            # Calculate physical area using pixel areas
            storm_pixels = mask & inside
            physical_area = np.sum(pixel_areas * storm_pixels)
            if physical_area >= area_threshold_km2:
                line, = ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
                storm_lines.append(line)
        return [img] + storm_lines
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=interval)
    return ani

def animate_storms_polar(data, storm_threshold=45, area_threshold_km2=10.0,
                         dilation_iterations=5, interval=100, figsize=(6, 6)):
    """
    Animate storm detection over time from radar reflectivity data in polar coordinates using physical area calculations.
    
    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - storm_threshold: dBZ threshold for storm detection (default: 45)
    - area_threshold_km2: minimum storm area in km² (default: 10.0)
    - dilation_iterations: dilation iterations for storm smoothing (default: 5)
    - interval: animation interval in milliseconds (default: 100)
    - figsize: figure size (default: (6, 6))
    """
    T, H, W = data.shape  # H=azimuth_bins, W=range_bins
    theta = np.linspace(0, 2 * np.pi, H, endpoint=False)  # Azimuth angles
    r = np.arange(W) * 0.5  # Range in km (500m intervals)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')  # (azimuth_bins, range_bins)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    ax.set_ylim(0, W * 0.5)  # Set range limit to maximum range in km
    norm = Normalize(vmin=0, vmax=80)
    cmap = plt.get_cmap("jet")
    quad = ax.pcolormesh(theta_grid, r_grid, data[0], cmap=cmap, norm=norm)
    title = ax.set_title("Radar Reflectivity - Time Step 0")
    storm_lines = []
    storm_patch = mpatches.Patch(color='red', label='Detected Storm')
    cbar = fig.colorbar(quad, ax=ax, pad=0.1)
    cbar.set_label('Reflectivity (dBZ)')
    ax.legend(handles=[storm_patch], loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # Compute pixel areas for the radar geometry
    pixel_areas = compute_polar_pixel_areas(data.shape[1:])
    
    def update(frame):
        nonlocal storm_lines
        frame_data = data[frame]
        quad.set_array(frame_data.ravel())
        title.set_text(f"Radar Reflectivity - Time Step {(frame*5)//60}:{frame*5%60:02d}")
        for line in storm_lines:
            line.remove()
        storm_lines = []
        mask = frame_data > storm_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
        contours = find_contours(dilated_mask.astype(float), 0.5)
        for contour in contours:
            # Use the same logic as the working animate_storms function
            path = Path(contour[:, ::-1])  # Same as working function
            
            # Create meshgrid - same as working function
            xg, yg = np.meshgrid(np.arange(W), np.arange(H))  # W=range_bins, H=azimuth_bins
            coords = np.vstack((xg.ravel(), yg.ravel())).T
            inside = path.contains_points(coords).reshape((H, W))  # (azimuth_bins, range_bins)
            
            # Calculate physical area using pixel areas
            storm_pixels = mask & inside
            physical_area = np.sum(pixel_areas * storm_pixels)
            if physical_area >= area_threshold_km2:
                # Convert contour coordinates to polar coordinates for plotting
                # Use the same logic as working function: contour[:, 1] is x, contour[:, 0] is y
                r_pts = contour[:, 1] * 0.5  # Convert range bins to km
                theta_pts = contour[:, 0] / H * 2 * np.pi  # Convert azimuth bins to radians
                line, = ax.plot(theta_pts, r_pts, color='red', linewidth=2)
                storm_lines.append(line)
        return [quad] + storm_lines
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    return ani

def animate_storms_polar_comparison(true_data, pred_data, storm_threshold=45, area_threshold_km2=10.0,
                                    dilation_iterations=5, interval=100, figsize=(12, 6)):
    """
    Animate storm detection comparison between true and predicted radar reflectivity data in polar coordinates using physical area calculations.
    
    Parameters:
    - true_data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - pred_data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - storm_threshold: dBZ threshold for storm detection (default: 45)
    - area_threshold_km2: minimum storm area in km² (default: 10.0)
    - dilation_iterations: dilation iterations for storm smoothing (default: 5)
    - interval: animation interval in milliseconds (default: 100)
    - figsize: figure size (default: (12, 6))
    """
    assert true_data.shape == pred_data.shape, "Input arrays must have the same shape."
    T, H, W = true_data.shape  # H=azimuth_bins, W=range_bins
    theta = np.linspace(0, 2*np.pi, H, endpoint=False)  # Azimuth angles
    r = np.arange(W) * 0.5  # Range in km (500m intervals)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')  # (azimuth_bins, range_bins)
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=figsize)
    for ax in axs:
        ax.set_ylim(0, W * 0.5)  # Set range limit to maximum range in km
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    titles = ['True Reflectivity', 'Predicted Reflectivity']
    norm = Normalize(vmin=0, vmax=80)
    cmap = plt.get_cmap("jet")
    quads = []
    storm_lines = [[], []]
    for i, ax in enumerate(axs):
        data = true_data if i == 0 else pred_data
        quad = ax.pcolormesh(theta_grid, r_grid, data[0], cmap=cmap, norm=norm)
        ax.set_title(f"{titles[i]} - Time Step 1")
        quads.append(quad)
    cbar = fig.colorbar(quads[0], ax=axs.ravel().tolist(), pad=0.1)
    cbar.set_label("Reflectivity (dBZ)")
    storm_patch = mpatches.Patch(color='red', label='Detected Storm')
    axs[1].legend(handles=[storm_patch], loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # Compute pixel areas for the radar geometry
    pixel_areas = compute_polar_pixel_areas(true_data.shape[1:])
    
    def update(frame):
        for i, ax in enumerate(axs):
            data = true_data if i == 0 else pred_data
            frame_data = data[frame]
            quads[i].set_array(frame_data.ravel())
            ax.set_title(f"{titles[i]} - Time Step {frame+1}")
            for line in storm_lines[i]:
                line.remove()
            storm_lines[i] = []
            mask = frame_data > storm_threshold
            dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
            contours = find_contours(dilated_mask.astype(float), 0.5)
            for contour in contours:
                # Use the same logic as the working animate_storms function
                path = Path(contour[:, ::-1])  # Same as working function
                
                # Create meshgrid - same as working function
                xg, yg = np.meshgrid(np.arange(W), np.arange(H))  # W=range_bins, H=azimuth_bins
                coords = np.vstack((xg.ravel(), yg.ravel())).T
                inside = path.contains_points(coords).reshape((H, W))  # (azimuth_bins, range_bins)
                
                # Calculate physical area using pixel areas
                storm_pixels = mask & inside
                physical_area = np.sum(pixel_areas * storm_pixels)
                if physical_area >= area_threshold_km2:
                    # Convert contour coordinates to polar coordinates for plotting
                    # Use the same logic as working function: contour[:, 1] is x, contour[:, 0] is y
                    r_pts = contour[:, 1] * 0.5  # Convert range bins to km
                    theta_pts = contour[:, 0] / H * 2 * np.pi  # Convert azimuth bins to radians
                    line, = ax.plot(theta_pts, r_pts, color='red', linewidth=2)
                    storm_lines[i].append(line)
        return quads + storm_lines[0] + storm_lines[1]
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    return ani

def animate_new_storms(data, new_storms_result):
    """
    Animate the progression of newly formed storms over time using radar reflectivity data.
    
    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - new_storms_result (list of dict): A list where each dict represents new storms at a specific time step.
        Each dict should have the keys:
        - "time_step" (int): Time step index.
        - "new_storm_count" (int): Number of new storms detected at that time step.
        - "new_storm_coordinates" (list of list of (x, y)): List of contours for each new storm.
        - "storm_areas_km2" (list of float): Physical areas of detected storms in km².
    Returns:
    - ani: matplotlib.animation.FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap("jet")
    img = ax.imshow(data[0], cmap=cmap, vmin=0, vmax=80)
    title = ax.set_title("New Storms at Time Step 0")
    colorbar = plt.colorbar(img, ax=ax, label="Reflectivity (dBZ)")
    storm_lines = []
    def update(frame_id):
        nonlocal storm_lines
        frame = data[frame_id]
        img.set_data(frame)
        title.set_text(f"New Storms at Time Step {frame_id}")
        for line in storm_lines:
            line.remove()
        storm_lines = []
        frame_entry = next((f for f in new_storms_result if f["time_step"] == frame_id), None)
        if frame_entry and frame_entry["new_storm_count"] > 0:
            for contour in frame_entry["new_storm_coordinates"]:
                contour = np.array(contour)
                line, = ax.plot(contour[:, 0], contour[:, 1], color='lime', linewidth=2)
                storm_lines.append(line)
        return [img, title] + storm_lines
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=200)
    return ani 

def animate_new_storms_with_wind(data, reflectivity_threshold=45, area_threshold_km2=10.0, 
                                dilation_iterations=5, overlap_threshold=0.1, interval=200,
                                patch_size=32, patch_stride=16, patch_thresh=35.0, patch_frac=0.015, 
                                maxv=85.0, use_high_reflectivity_patches=True):
    """
    Animate new storm detection with displacement-based prediction visualization.
    
    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - reflectivity_threshold: dBZ threshold for storm detection (default: 45)
    - area_threshold_km2: minimum storm area in km² (default: 10.0)
    - dilation_iterations: dilation iterations for storm smoothing (default: 5)
    - overlap_threshold: overlap threshold for new storm detection (default: 0.1)
    - interval: animation interval in milliseconds (default: 200)
    - patch_size: int, size of patches for cross-correlation (default: 32)
    - patch_stride: int, stride between patches (default: 16)
    - patch_thresh: float, threshold for patch selection in dBZ (default: 35.0)
    - patch_frac: float, minimum fraction of pixels above threshold (default: 0.015)
    - maxv: float, maximum value for normalization (default: 85.0)
    - use_high_reflectivity_patches: bool, whether to only use patches with high reflectivity (default: True)
    
    Returns:
    - ani: matplotlib.animation.FuncAnimation object
    """

    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap("jet")
    img = ax.imshow(data[0], cmap=cmap, vmin=0, vmax=80, extent=[0, data.shape[2], data.shape[1], 0])
    title = ax.set_title("Displacement-Based New Storm Detection - Time Step 0")
    colorbar = plt.colorbar(img, ax=ax, label="Reflectivity (dBZ)")
    
    # Set proper axis limits 
    ax.set_xlim(0, data.shape[2])
    ax.set_ylim(data.shape[1], 0)
    
    # Get all storm results for prediction
    storm_results = detect_storms(data, reflectivity_threshold, area_threshold_km2, dilation_iterations)
    
    # Get new storm formations (this will compute displacement vectors internally)
    result = detect_new_storm_formations(
        data, reflectivity_threshold, area_threshold_km2, 
        dilation_iterations, overlap_threshold, use_displacement_prediction=True,
        patch_size=patch_size, patch_stride=patch_stride,
        patch_thresh=patch_thresh, patch_frac=patch_frac, maxv=maxv,
        use_high_reflectivity_patches=use_high_reflectivity_patches
    )
    
    # Check if displacement fields and patch centers were returned
    if isinstance(result, tuple) and len(result) == 3:
        new_storms_result, displacement_fields, selected_patch_centers = result
    elif isinstance(result, tuple) and len(result) == 2:
        new_storms_result, displacement_fields = result
        # If no patch centers returned, compute them for visualization
        _, _, selected_patch_centers = compute_displacement_vectors(
            data, patch_size=patch_size, patch_stride=patch_stride,
            patch_thresh=patch_thresh, patch_frac=patch_frac, maxv=maxv,
            use_high_reflectivity_patches=use_high_reflectivity_patches,
            show_progress=False
        )
    else:
        new_storms_result = result
        # If no displacement fields returned, compute them for visualization
        _, displacement_fields, selected_patch_centers = compute_displacement_vectors(
            data, patch_size=patch_size, patch_stride=patch_stride,
            patch_thresh=patch_thresh, patch_frac=patch_frac, maxv=maxv,
            use_high_reflectivity_patches=use_high_reflectivity_patches,
            show_progress=False
        )
    
    storm_lines = []
    predicted_lines = []
    new_storm_lines = []
    wind_arrows = []
    
    def update(frame_id):
        nonlocal storm_lines, predicted_lines, new_storm_lines, wind_arrows
        
        # Clear previous lines and arrows
        for line in storm_lines + predicted_lines + new_storm_lines:
            line.remove()
        for arrow in wind_arrows:
            arrow.remove()
        storm_lines = []
        predicted_lines = []
        new_storm_lines = []
        wind_arrows = []
        
        frame = data[frame_id]
        img.set_data(frame)
        title.set_text(f"Displacement-Based New Storm Detection - Time Step {frame_id}")
        
        # Get current storms
        current_storms = storm_results[frame_id]
        current_masks = current_storms['storm_masks']
        current_coords = current_storms['storm_coordinates']
        
        # Plot all current storms in red (same as other plots)
        for contour in current_coords:
            contour = np.array(contour)
            line, = ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)
            storm_lines.append(line)
        
        # Plot predicted storm positions if not first frame
        if frame_id > 0 and len(storm_results[frame_id-1]['storm_masks']) > 0:
            displacement_field = displacement_fields[frame_id-1]
            previous_masks = storm_results[frame_id-1]['storm_masks']
            predicted_masks = predict_storm_positions(previous_masks, displacement_field, data.shape[1:])
            
            # Create contours for predicted masks
            for pred_mask in predicted_masks:
                contours = find_contours(pred_mask.astype(float), 0.5)
                for contour in contours:
                    line, = ax.plot(contour[:, 1], contour[:, 0], color='orange', 
                                  linewidth=2, linestyle='--')
                    predicted_lines.append(line)
        
        # Plot new storms in bright green 
        frame_entry = next((f for f in new_storms_result if f["time_step"] == frame_id), None)
        if frame_entry and frame_entry["new_storm_count"] > 0:
            for contour in frame_entry["new_storm_coordinates"]:
                contour = np.array(contour)
                line, = ax.plot(contour[:, 0], contour[:, 1], color='lime', linewidth=2)
                new_storm_lines.append(line)
        
        # Plot displacement vectors as arrows on a global grid, using real displacement in patches and dummy arrows elsewhere
        if frame_id > 0 and frame_id-1 < len(selected_patch_centers):
            displacement_field = displacement_fields[frame_id-1]
            patch_centers = selected_patch_centers[frame_id-1]
            step = 5  # Arrow grid density
            scale_factor = 4.5  # Make arrows 4.5 times longer than the actual displacement
            min_arrow_length = 3.0
            
            # Create a mask of all pixels covered by any selected patch
            # IMPORTANT: Use the SAME patch mask creation logic as the diagnostic section
            patch_mask = np.zeros((data.shape[1], data.shape[2]), dtype=bool)
            for center_y, center_x in patch_centers:
                y_start = max(center_y - patch_size // 2, 0)
                x_start = max(center_x - patch_size // 2, 0)
                y_end = min(y_start + patch_size, data.shape[1])
                x_end = min(x_start + patch_size, data.shape[2])
                patch_mask[y_start:y_end, x_start:x_end] = True
            

            

            

            # Plot displacement vectors as arrows on a global grid
            # Real arrows in patch regions, dummy arrows elsewhere
            step = 15  # Arrow grid density
            scale_factor = 4.5  # Make arrows 4.5 times longer than the actual displacement
            min_arrow_length = 3.0
            
            # Loop over a global grid - ensure equal gaps on all borders
            # Calculate the total number of steps that fit in the image
            y_steps = (data.shape[1] - 1) // step
            x_steps = (data.shape[2] - 1) // step
            
            # Calculate the starting positions to center the grid
            y_start = (data.shape[1] - 1 - (y_steps * step)) // 2
            x_start = (data.shape[2] - 1 - (x_steps * step)) // 2
            
            for y in range(y_start, data.shape[1], step):
                for x in range(x_start, data.shape[2], step):
                    if patch_mask[y, x]:
                        # Real displacement arrow
                        u = displacement_field[y, x, 0]
                        v = displacement_field[y, x, 1]
                        u_scaled = u * scale_factor
                        v_scaled = v * scale_factor
                        if abs(u_scaled) < min_arrow_length:
                            u_scaled = np.sign(u_scaled) * min_arrow_length
                        if abs(v_scaled) < min_arrow_length:
                            v_scaled = np.sign(v_scaled) * min_arrow_length
                        # Clip arrow tips to stay within image bounds
                        max_u = data.shape[2] - 1 - x  # Maximum u that keeps arrow within bounds
                        max_v = data.shape[1] - 1 - y  # Maximum v that keeps arrow within bounds
                        min_u = -x  # Minimum u that keeps arrow within bounds
                        min_v = -y  # Minimum v that keeps arrow within bounds
                        
                        # Clip the scaled components to stay within bounds
                        u_scaled_clipped = np.clip(u_scaled, min_u, max_u)
                        v_scaled_clipped = np.clip(v_scaled, min_v, max_v)
                        
                        # Always plot the arrow (now guaranteed to be within bounds)
                        arrow = ax.arrow(x, y, u_scaled_clipped, v_scaled_clipped,
                                         head_width=1.5, head_length=1.5,
                                         fc='red', ec='red', alpha=0.8, linewidth=1)
                        wind_arrows.append(arrow)
                    else:
                        # Dummy arrow: just an arrowhead marker, no tail
                        marker = ax.plot(x, y, marker='>', color='red', markersize=1.5, alpha=0.8, linewidth=0)[0]
                        wind_arrows.append(marker)
                        

        
        # Add legend
        if frame_id == 0:  # Only add legend once
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Storms'),
                Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Predicted Positions'),
                Line2D([0], [0], color='lime', lw=2, label='New Storms'),
                Line2D([0], [0], color='red', lw=1, marker='>', markersize=8, label='Displacement Vectors')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        

        
        return [img, title] + storm_lines + predicted_lines + new_storm_lines + wind_arrows
    
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=interval)
    return ani 
