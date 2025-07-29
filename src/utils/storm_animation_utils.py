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
                                dilation_iterations=5, overlap_threshold=0.1, interval=200):
    """
    Animate new storm detection with wind-based prediction visualization.
    
    Parameters:
    - data: np.ndarray of shape (T, H, W) where H=azimuth_bins, W=range_bins
    - reflectivity_threshold: dBZ threshold for storm detection (default: 45)
    - area_threshold_km2: minimum storm area in km² (default: 10.0)
    - dilation_iterations: dilation iterations for storm smoothing (default: 5)
    - overlap_threshold: overlap threshold for new storm detection (default: 0.1)
    - interval: animation interval in milliseconds (default: 200)

    
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
        dilation_iterations, overlap_threshold, use_displacement_prediction=True
    )
    
    # Check if displacement fields were returned
    if isinstance(result, tuple):
        new_storms_result, displacement_fields = result
    else:
        new_storms_result = result
        # If no displacement fields returned, compute them for visualization
        displacement_vectors, displacement_fields = compute_displacement_vectors(data, show_progress=False)
    
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
        
        # Plot displacement vectors as arrows 
        if frame_id > 0:  # Only show displacement vectors after first frame
            displacement_field = displacement_fields[frame_id-1]
            

            
            # Create a grid for displacement vectors (every 25th pixel to reduce clutter)
            step = 25
            y_grid, x_grid = np.mgrid[step//2:data.shape[1]:step, step//2:data.shape[2]:step]
            
            # Get displacement vectors at grid points
            u_vals = displacement_field[y_grid, x_grid, 0]
            v_vals = displacement_field[y_grid, x_grid, 1]
            

            
            # Scale displacement vectors to make them more visible (4.5x longer than actual displacement)
            # The displacement vectors represent pixel displacement from cross-correlation
            scale_factor = 4.5  # Make arrows 4.5 times longer for better visibility
            
            u_scaled = u_vals * scale_factor
            v_scaled = v_vals * scale_factor
            
            # Ensure minimum arrow size for visibility 
            min_arrow_length = 3.0
            u_scaled = np.where(np.abs(u_scaled) < min_arrow_length, 
                              np.sign(u_scaled) * min_arrow_length, u_scaled)
            v_scaled = np.where(np.abs(v_scaled) < min_arrow_length, 
                              np.sign(v_scaled) * min_arrow_length, v_scaled)
            
            # Plot displacement vectors as arrows showing predicted pixel displacement
            for i in range(x_grid.shape[0]):
                for j in range(x_grid.shape[1]):
                    x, y = x_grid[i, j], y_grid[i, j]
                    u, v = u_scaled[i, j], v_scaled[i, j]
                    
                    # Only plot if the predicted position is within bounds
                    predicted_x = x + u
                    predicted_y = y + v
                    if 0 <= predicted_x < data.shape[2] and 0 <= predicted_y < data.shape[1]:
                        # Draw arrow from current position to predicted position
                        # The arrow should point in the direction of movement
                        arrow = ax.arrow(x, y, u, v, 
                                       head_width=1.5, head_length=1.5, 
                                       fc='red', ec='red', alpha=0.8, linewidth=1)
                        wind_arrows.append(arrow)
        
        # Add legend
        if frame_id == 0:  # Only add legend once
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='All Storms'),
                Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Predicted Positions'),
                Line2D([0], [0], color='lime', lw=2, label='New Storms'),
                Line2D([0], [0], color='red', lw=1, marker='>', markersize=8, label='Displacement Vectors')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        return [img, title] + storm_lines + predicted_lines + new_storm_lines + wind_arrows
    
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=interval)
    return ani 
