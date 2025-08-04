import os
import numpy as np
import json
from tqdm import tqdm
import argparse

def join_data(input_dir, output_dir, output_name):
    """Join processed data from intermediate directory into final dataset."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Joining data from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output filename: {output_name}")
    
    data_arrays = []
    all_filenames = []

    # First, collect all relevant directories for progress bar
    join_targets = []   

    for root, dirs, files in os.walk(input_dir):      # for data in intermediate directory
        dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
        files.sort()
        if 'data.npy' in files and 'filenames.json' in files:
            join_targets.append(root)

    if not join_targets:
        print(f"No processed data found in {input_dir}")
        return

    # Determine total number of samples and sample shape
    total_samples = 0
    sample_shape = None
    for root in join_targets:
        arr = np.load(os.path.join(root, 'data.npy'), mmap_mode='r')
        if sample_shape is None:
            sample_shape = arr.shape[1:]  
        total_samples += arr.shape[0]

    # Pre-allocate memmap array for output
    out_path = os.path.join(output_dir, output_name)
    final_data = np.lib.format.open_memmap(out_path, mode='w+', dtype='float32', shape=(total_samples, *sample_shape))

    # Fill the memmap array and join filenames
    idx = 0
    all_filenames = []
    for root in tqdm(join_targets, desc="Joining processed data"):
        arr = np.load(os.path.join(root, 'data.npy'))
        n = arr.shape[0]
        final_data[idx:idx+n] = arr
        idx += n
        with open(os.path.join(root, 'filenames.json')) as f:
            names = json.load(f)
            all_filenames.extend(names)

    # Always save filenames as ZH_radar_filenames.json
    filenames_path = os.path.join(output_dir, 'ZH_radar_filenames.json')
    with open(filenames_path, 'w') as f:
        json.dump(all_filenames, f)
    
    print(f"Saved concatenated data to {out_path}, shape: {final_data.shape}")
    print(f"Saved concatenated filenames to {filenames_path}, count: {len(all_filenames)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Join processed radar data into final dataset')
    parser.add_argument('--input_dir', type=str, default="data/intermediate",
                       help='Input directory containing processed data (default: data/intermediate)')
    parser.add_argument('--output_dir', type=str, default="data/processed",
                       help='Output directory for final dataset (default: data/processed)')
    parser.add_argument('--output_name', type=str, default="ZH_radar_dataset_raw.npy",
                       help='Output filename for the dataset (default: ZH_radar_dataset_raw.npy)')
    args = parser.parse_args()
    
    join_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name
    ) 