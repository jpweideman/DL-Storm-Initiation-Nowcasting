import os
import numpy as np
import wradlib as wrl
from tqdm import tqdm  
import json
import argparse

# Set WRADLIB_DATA folder
data_dir = "data/raw"
os.environ['WRADLIB_DATA'] = data_dir

# Output root directory for intermediate processed data
output_root = "data/intermediate"
os.makedirs(output_root, exist_ok=True)

# Helper function to process one file
def process_one_file(file_path, target_h, target_w, num_channels=14, variable="ZH", noise_value=96.00197):
    """
    Process a single HDF5 file to extract radar data.
    
    Args:
        file_path: Path to the HDF5 file
        target_h: Target height for output arrays
        target_w: Target width for output arrays
        num_channels: Number of channels to process (default: 14)
        variable: Variable to extract (default: ZH)
        noise_value: Value to replace with 0 (noise cleaning)
    
    Returns:
        numpy array of shape (num_channels, target_h, target_w)
    """
    data, _ = wrl.io.read_gamic_hdf5(file_path)
    processed_scans = []
    for i in tqdm(range(num_channels), desc=f"Scans in {os.path.basename(file_path)}", leave=False):
        scan_key = f"SCAN{i}"
        if variable not in data[scan_key]:
            raise ValueError(f"{variable} not found in {scan_key} of file {file_path}")
        arr = data[scan_key][variable]["data"]
        arr[arr == noise_value] = 0  # clean noise
        h, w = arr.shape
        # Pad height
        if h > target_h:
            arr = arr[:target_h, :]
        elif h < target_h:
            pad_bottom = target_h - h
            arr = np.pad(arr, ((0, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        # Pad width
        if w > target_w:
            arr = arr[:, :target_w]
        elif w < target_w:
            pad_right = target_w - w
            arr = np.pad(arr, ((0, 0), (0, pad_right)), mode='constant', constant_values=0)
        processed_scans.append(arr)
    return np.stack(processed_scans)  # shape: (num_channels, target_h, target_w)

def main():
    """Main processing function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process radar HDF5 files to numpy arrays')
    parser.add_argument('--target_height', type=int, default=360,
                       help='Target height for output arrays (default: 360)')
    parser.add_argument('--target_width', type=int, default=240,
                       help='Target width for output arrays (default: 240)')
    parser.add_argument('--num_channels', type=int, default=14,
                       help='Number of channels/scans to process (default: 14)')
    parser.add_argument('--variable', type=str, default="ZH",
                       help='Variable to extract from scans (default: ZH)')
    parser.add_argument('--noise_value', type=float, default=96.00197,
                       help='Noise value to replace with 0 (default: 96.00197)')
    args = parser.parse_args()
    
    print(f"Processing data from: {data_dir}")
    print(f"Output directory: {output_root}")
    print(f"Target shape: {args.target_height}x{args.target_width}")
    print(f"Number of channels: {args.num_channels}")
    print(f"Variable: {args.variable}")
    print(f"Noise value: {args.noise_value}")
    print("-" * 50)
    
    # Recursively process and save by directory
    for root, dirs, files in os.walk(data_dir):
        # Sort year and month directories numerically if possible
        dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
        files.sort()
        h5_files = [f for f in files if f.endswith(".h5")]
        if not h5_files:
            continue
        # Create corresponding output directory
        rel_dir = os.path.relpath(root, data_dir)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_npy = os.path.join(out_dir, "data.npy")
        if os.path.exists(out_npy):
            print(f"Skipping {out_dir} (already processed)")
            continue
        tensors = []
        rel_filenames = []
        for fname in tqdm(sorted(h5_files), desc=f"Processing {rel_dir}"):
            fpath = os.path.join(root, fname)
    try:
                tensor = process_one_file(
                    fpath, 
                    args.target_height, 
                    args.target_width, 
                    args.num_channels,
                    args.variable,
                    args.noise_value
                )
                tensors.append(tensor)
                rel_filenames.append(os.path.relpath(fpath, data_dir))
    except Exception as e:
                print(f"Error processing {fpath}: {e}")
                if tensors:
                    np.save(out_npy, np.stack(tensors))
                    with open(os.path.join(out_dir, "filenames.json"), "w") as f:
                        json.dump(rel_filenames, f)
                    print(f"Saved {len(tensors)} tensors to {out_npy}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()