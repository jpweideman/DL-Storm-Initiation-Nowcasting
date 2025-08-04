import argparse
import numpy as np
import os
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from tqdm import tqdm


def calculate_height_agl(range_km: np.ndarray, elevation_deg: np.ndarray, 
                        radar_height_km: float = 0.0) -> np.ndarray:
    """
    Calculate height above ground level for radar data using the 4/3 Earth radius model.
    
    Uses the exact formula from radar meteorology literature:
    h = √((R_eff + h_0)² + r² + 2(R_eff + h_0)r sin(ε_0)) - R_eff
    
    Parameters
    ----------
    range_km : np.ndarray
        Range values in kilometers.
    elevation_deg : np.ndarray
        Elevation angles in degrees.
    radar_height_km : float, optional
        Height of radar antenna above ground in kilometers (default: 0.0).
    
    Returns
    -------
    np.ndarray
        Height above ground level in kilometers.
    """
    # Convert elevation to radians
    elevation_rad = np.radians(elevation_deg)
    
    # Earth's radius in km
    earth_radius = 6371.0
    
    # Effective Earth radius using 4/3 Earth radius model
    effective_earth_radius = 4/3 * earth_radius
    
    # Exact formula from radar meteorology literature
    # h = √((R_eff + h_0)² + r² + 2(R_eff + h_0)r sin(ε_0)) - R_eff
    height_agl = np.sqrt(
        (effective_earth_radius + radar_height_km)**2 + 
        range_km**2 + 
        2 * (effective_earth_radius + radar_height_km) * range_km * np.sin(elevation_rad)
    ) - effective_earth_radius
    
    return height_agl


def create_ground_clutter_mask(range_km: np.ndarray, elevation_deg: np.ndarray, 
                              clutter_height_km: float = 1.0, 
                              radar_height_above_ground_m: float = 0.0) -> np.ndarray:
    """
    Create ground clutter mask for radar data.
    
    Parameters
    ----------
    range_km : np.ndarray
        Range values in kilometers (1D array).
    elevation_deg : np.ndarray
        Elevation angles in degrees (1D array).
    clutter_height_km : float, optional
        Height above ground level below which to mask data (default: 1.0).
    radar_height_above_ground_m : float, optional
        Height of radar antenna above ground in meters (default: 0.0).
    
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates valid data (above clutter height).
        Shape: (len(elevation_deg), len(range_km))
    """
    range_grid, elev_grid = np.meshgrid(range_km, elevation_deg)
    
    radar_height_km = radar_height_above_ground_m / 1000.0
    
    # Calculate height AGL for each pixel
    height_agl = calculate_height_agl(range_grid, elev_grid, radar_height_km)
    
    # Mask is True where height > clutter_height
    mask = height_agl > clutter_height_km
    
    return mask


def remove_ground_clutter_chunked(radar_data: np.ndarray, range_km: np.ndarray, 
                                 elevation_deg: np.ndarray, clutter_height_km: float = 1.0,
                                 radar_height_above_ground_m: float = 0.0, chunk_size: int = 100,
                                 output_file: str = None) -> np.ndarray:
    """
    Remove ground clutter from radar data using chunked processing for memory efficiency.
    
    Supports 4D data with shape (time, elevation, azimuth, range).
    
    Parameters
    ----------
    radar_data : np.ndarray
        Radar reflectivity data. Shape: (time, elevation, azimuth, range).
    range_km : np.ndarray
        Range values in kilometers (1D array).
    elevation_deg : np.ndarray
        Elevation angles in degrees (1D array).
    clutter_height_km : float, optional
        Height above ground level below which to set data to 0 (default: 1.0).
    radar_height_above_ground_m : float, optional
        Height of radar antenna above ground in meters (default: 0.0).
    chunk_size : int, optional
        Number of time steps to process at once (default: 100).
    output_file : str
        Output file path for memory-mapped array.
    
    Returns
    -------
    np.ndarray
        Radar data with ground clutter removed (set to 0).
    """
    clutter_mask = create_ground_clutter_mask(range_km, elevation_deg, 
                                             clutter_height_km, radar_height_above_ground_m)
    
    total_time_steps = radar_data.shape[0]
    
    cleaned_data = np.lib.format.open_memmap(output_file, mode='w+', dtype='float32', 
                                            shape=radar_data.shape)
    
    for start_idx in tqdm(range(0, total_time_steps, chunk_size), 
                         desc="Removing ground clutter", unit="chunk"):
        end_idx = min(start_idx + chunk_size, total_time_steps)
        
        chunk = radar_data[start_idx:end_idx]  
        
        chunk_mask = np.expand_dims(clutter_mask, 0)  
        chunk_mask = np.expand_dims(chunk_mask, 2)    
        chunk_mask = np.repeat(chunk_mask, chunk.shape[0], axis=0)  
        chunk_mask = np.repeat(chunk_mask, chunk.shape[2], axis=2)  
        
        cleaned_chunk = chunk * chunk_mask
        
        cleaned_data[start_idx:end_idx] = cleaned_chunk
    
    return cleaned_data


def create_range_array(max_range_km: float, range_resolution_m: float) -> np.ndarray:
    """
    Create range array for radar data.
    
    Parameters
    ----------
    max_range_km : float
        Maximum range in kilometers.
    range_resolution_m : float
        Range resolution in meters.
    
    Returns
    -------
    np.ndarray
        Array of range values in kilometers.
    """
    range_resolution_km = range_resolution_m / 1000.0
    return np.arange(range_resolution_km, max_range_km + range_resolution_km, range_resolution_km)


def parse_elevations(elevations_str: str) -> List[float]:
    """
    Parse elevation angles from string.
    
    Parameters
    ----------
    elevations_str : str
        Comma-separated string of elevation angles.
    
    Returns
    -------
    List[float]
        List of elevation angles in degrees.
    """
    return [float(x.strip()) for x in elevations_str.split(',')]


def main():
    """Main function for removing ground clutter from radar data."""
    parser = argparse.ArgumentParser(
        description="Remove ground clutter from radar data using height-based masking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input_file', type=str, default="data/processed/ZH_radar_dataset_raw.npy",
                       help='Path to input radar data file (.npy, default: data/processed/ZH_radar_dataset_raw.npy)')
    parser.add_argument('--output_file', type=str, default="data/processed/ZH_radar_dataset.npy",
                       help='Path to output cleaned radar data file (.npy, default: data/processed/ZH_radar_dataset.npy)')
    parser.add_argument('--clutter_height', type=float, default=1.0,
                       help='Height above ground level below which to set data to 0 (km, default: 1.0)')
    parser.add_argument('--radar_height_above_ground', type=float, default=38.0,
                       help='Height of radar antenna above ground (m, default: 38.0 for KITradar)')
    parser.add_argument('--elevations', type=str, 
                       default='0.4,1.1,2.0,3.0,4.5,6.0,7.5,9.0,11.0,13.0,16.0,20.0,24.0,30.0',
                       help='Comma-separated list of elevation angles in degrees (default: KITradar elevations)')
    parser.add_argument('--max_range', type=float, default=120.0,
                       help='Maximum range in kilometers (default: 120.0)')
    parser.add_argument('--range_resolution', type=float, default=500.0,
                       help='Range resolution in meters (default: 500.0)')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Number of time steps to process at once (default: 100)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse elevations
    try:
        elevation_deg = parse_elevations(args.elevations)
    except ValueError as e:
        print(f"Error parsing elevations: {e}")
        sys.exit(1)
    
    # Create range array
    range_km = create_range_array(args.max_range, args.range_resolution)
    
    # Load radar data 
    print(f"Loading radar data from: {args.input_file}")
    try:
        radar_data = np.load(args.input_file, mmap_mode='r')
    except Exception as e:
        print(f"Error loading radar data: {e}")
        sys.exit(1)
    
    # Check data shape
    if len(radar_data.shape) != 4:
        print(f"Error: Expected 4D array (time, elevation, azimuth, range), got shape {radar_data.shape}")
        sys.exit(1)
    
    time_steps, num_elevations, num_azimuth, num_range_bins = radar_data.shape
    
    if num_elevations != len(elevation_deg):
        print(f"Error: Number of elevations in data ({num_elevations}) doesn't match provided elevations ({len(elevation_deg)})")
        sys.exit(1)
    
    print(f"Data shape: {radar_data.shape}")
    print(f"Time steps: {time_steps}")
    print(f"Elevations: {len(elevation_deg)}")
    print(f"Azimuth angles: {num_azimuth}")
    print(f"Range bins: {num_range_bins}")
    print(f"Clutter height threshold: {args.clutter_height} km above ground")
    print(f"Radar height: {args.radar_height_above_ground} m above ground")
    
    # Remove ground clutter with memory mapping
    cleaned_data = remove_ground_clutter_chunked(
        radar_data, range_km, elevation_deg, 
        clutter_height_km=args.clutter_height,
        radar_height_above_ground_m=args.radar_height_above_ground,
        chunk_size=args.chunk_size,
        output_file=args.output_file
    )
    

    print(f"Cleaned data saved to: {args.output_file}")


if __name__ == "__main__":
    main() 