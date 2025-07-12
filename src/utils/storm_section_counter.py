import numpy as np
import argparse
import json
import os
from storm_utils import detect_storms, detect_new_storm_formations
from tqdm import tqdm

def count_storms_by_section(data, interval_percent=5, batch_size=10, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5, overlap_threshold=0.1):
    """
    Count the number of storms and new storms in each section of the data, processing each section in small, fixed-size batches.
    Parameters:
        data: np.ndarray, shape (T, H, W) or (N, C, H, W) (can be memmap)
        interval_percent: int, percentage size of each section (e.g., 5 for 5%)
        batch_size: int, number of frames to process at a time
        reflectivity_threshold, area_threshold, dilation_iterations: passed to detect_storms
        overlap_threshold: float, passed to detect_new_storm_formations
    Returns:
        List of dicts: [{"section": 1, "start": 0, "end": 10, "storm_count": 20, "new_storm_count": 5}, ...]
    """
    T = data.shape[0]
    section_size = max(1, int(np.ceil(T * interval_percent / 100)))
    n_sections = int(np.ceil(T / section_size))
    sections = []
    prev_section_masks = []  # For new storm detection across section boundaries
    for sec_idx, i in enumerate(tqdm(range(0, T, section_size), desc="Processing sections", total=n_sections)):
        start = i
        end = min(i + section_size, T)
        total_storms = 0
        total_new_storms = 0
        
        # Process section in batches for both storm counting and new storm detection
        batch_iter = range(start, end, batch_size)
        current_section_masks = []  # Collect masks from this section for next section
        
        for batch_start in tqdm(batch_iter, desc=f"Section {sec_idx+1}/{n_sections} batches", leave=False):
            batch_end = min(batch_start + batch_size, end)
            batch_data = data[batch_start:batch_end]
            if batch_data.ndim == 4:
                batch_data = np.max(batch_data, axis=1)
            
            # Count storms in this batch
            storms = detect_storms(batch_data, reflectivity_threshold, area_threshold, dilation_iterations)
            total_storms += sum(frame["storm_count"] for frame in storms)
            
            # Count new storms in this batch
            if prev_section_masks and batch_start == start:
                # First batch of section: check against previous section's masks
                batch_new_storms = 0
                for frame_result in storms:
                    new_count = 0
                    current_masks = frame_result['storm_masks']
                    for current in current_masks:
                        is_new = True
                        for prev in prev_section_masks:
                            overlap = np.sum(current & prev) / np.sum(current)
                            if overlap > overlap_threshold:
                                is_new = False
                                break
                        if is_new:
                            new_count += 1
                    batch_new_storms += new_count
                total_new_storms += batch_new_storms
            else:
                # Use detect_new_storm_formations for subsequent batches or first section
                if batch_start == start and not prev_section_masks:
                    # First section, first batch
                    new_storms = detect_new_storm_formations(batch_data, reflectivity_threshold, area_threshold, dilation_iterations, overlap_threshold)
                else:
                    # Subsequent batches: need to check against previous batch's masks
                    if batch_start == start:
                        # First batch of section but not first section
                        # Create a dummy frame with previous masks for continuity
                        dummy_data = np.zeros_like(batch_data[:1])
                        combined_data = np.concatenate([dummy_data, batch_data], axis=0)
                        new_storms = detect_new_storm_formations(combined_data, reflectivity_threshold, area_threshold, dilation_iterations, overlap_threshold)
                        # Remove the dummy frame count
                        total_new_storms += sum(frame['new_storm_count'] for frame in new_storms[1:])
                    else:
                        # Regular batch: check against previous batch's masks
                        # Get previous batch's masks
                        prev_batch_start = batch_start - batch_size
                        prev_batch_end = batch_start
                        prev_batch_data = data[prev_batch_start:prev_batch_end]
                        if prev_batch_data.ndim == 4:
                            prev_batch_data = np.max(prev_batch_data, axis=1)
                        prev_batch_storms = detect_storms(prev_batch_data, reflectivity_threshold, area_threshold, dilation_iterations)
                        prev_batch_masks = []
                        for frame_result in prev_batch_storms:
                            prev_batch_masks.extend(frame_result['storm_masks'])
                        
                        # Check current batch against previous batch masks
                        batch_new_storms = 0
                        for frame_result in storms:
                            new_count = 0
                            current_masks = frame_result['storm_masks']
                            for current in current_masks:
                                is_new = True
                                for prev in prev_batch_masks:
                                    overlap = np.sum(current & prev) / np.sum(current)
                                    if overlap > overlap_threshold:
                                        is_new = False
                                        break
                                if is_new:
                                    new_count += 1
                            batch_new_storms += new_count
                        total_new_storms += batch_new_storms
                        continue  # Skip the else block below
            
            # Collect masks from this batch for next section
            for frame_result in storms:
                current_section_masks.extend(frame_result['storm_masks'])
        
        # Save last masks for next section
        prev_section_masks = current_section_masks
        
        sections.append({
            "section": sec_idx+1,
            "start": start,
            "end": end,
            "storm_count": total_storms,
            "new_storm_count": total_new_storms
        })
    return sections

def main():
    parser = argparse.ArgumentParser(description="Count storms in sections of radar data using storm_utils.")
    parser.add_argument('--npy_path', type=str, required=True, help='Path to radar data .npy file (T,H,W or N,C,H,W)')
    parser.add_argument('--interval_percent', type=int, default=5, help='Section interval as percent of data length (default: 5)')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of frames to process at a time (default: 10)')
    parser.add_argument('--reflectivity_threshold', type=float, default=45, help='Reflectivity threshold (default: 45)')
    parser.add_argument('--area_threshold', type=int, default=15, help='Minimum storm area (default: 15)')
    parser.add_argument('--dilation_iterations', type=int, default=5, help='Dilation iterations (default: 5)')
    parser.add_argument('--overlap_threshold', type=float, default=0.1, help='Overlap threshold for new storm detection (default: 0.1)')
    parser.add_argument('--out', type=str, default=None, help='Optional output JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        raise FileNotFoundError(f"File not found: {args.npy_path}")
    print(f"Starting storm section counter on {args.npy_path}")
    data = np.load(args.npy_path, mmap_mode='r')
    results = count_storms_by_section(
        data,
        interval_percent=args.interval_percent,
        batch_size=args.batch_size,
        reflectivity_threshold=args.reflectivity_threshold,
        area_threshold=args.area_threshold,
        dilation_iterations=args.dilation_iterations,
        overlap_threshold=args.overlap_threshold
    )
    print("\nStorm and new storm counts by section:")
    for section in results:
        print(f"Section {section['section']}: Frames {section['start']}–{section['end']} → {section['storm_count']} storms, {section['new_storm_count']} new storms")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main() 