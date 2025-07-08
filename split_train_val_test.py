import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Split a radar dataset into train/val and test sets (chronologically)")
parser.add_argument('--input_npy', type=str, required=True, help='Path to input .npy radar file (shape: T, C, H, W)')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save split .npy files')
parser.add_argument('--test_frac', type=float, default=0.1, help='Fraction of data to hold out for test set (default: 0.1)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (not used, for compatibility)')
args = parser.parse_args()

input_npy = Path(args.input_npy)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

cube = np.load(input_npy, mmap_mode='r')
T = cube.shape[0]
test_size = int(T * args.test_frac)
train_val_size = T - test_size

# Chronological split
train_val = cube[:train_val_size]
test = cube[train_val_size:]

# Save with progress bar if large
chunk_size = 1000  # adjust as needed

def save_with_progress(array, out_path, desc):
    if array.shape[0] <= chunk_size:
        np.save(out_path, array)
    else:
        # Save in chunks
        memmap = np.lib.format.open_memmap(out_path, mode='w+', dtype=array.dtype, shape=array.shape)
        for i in tqdm(range(0, array.shape[0], chunk_size), desc=desc):
            memmap[i:i+chunk_size] = array[i:i+chunk_size]
        del memmap  # flush to disk

save_with_progress(train_val, output_dir / 'train_val.npy', desc='Saving train_val.npy')
save_with_progress(test, output_dir / 'test.npy', desc='Saving test.npy')

# Also split the filenames json if it exists
json_path = input_npy.parent / (input_npy.stem.replace('_dataset', '_filenames') + '.json')
if json_path.exists():
    with open(json_path, 'r') as f:
        filenames = json.load(f)
    assert len(filenames) == T, f"Filenames length {len(filenames)} does not match data length {T}"
    train_val_filenames = filenames[:train_val_size]
    test_filenames = filenames[train_val_size:]
    # Save filenames with progress if large
    def save_json_with_progress(data, out_path, desc):
        if len(data) <= chunk_size:
            with open(out_path, 'w') as f:
                json.dump(data, f)
        else:
            # Write in chunks for progress
            with open(out_path, 'w') as f:
                f.write('[')
                for i in tqdm(range(0, len(data), chunk_size), desc=desc):
                    chunk = data[i:i+chunk_size]
                    json.dump(chunk, f)
                    if i + chunk_size < len(data):
                        f.write(',')
                f.write(']')
    save_json_with_progress(train_val_filenames, output_dir / 'train_val_filenames.json', desc='Saving train_val_filenames.json')
    save_json_with_progress(test_filenames, output_dir / 'test_filenames.json', desc='Saving test_filenames.json')
    print(f"Train/Val filenames: {len(train_val_filenames)}")
    print(f"Test filenames: {len(test_filenames)}")
else:
    print(f"Warning: {json_path} not found, skipping filenames split.")

print(f"Total samples: {T}")
print(f"Train/Val samples: {train_val.shape[0]}")
print(f"Test samples: {test.shape[0]}")
print(f"Saved to {output_dir}") 