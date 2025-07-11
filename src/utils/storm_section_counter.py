import numpy as np
import argparse
import json
import os
from storm_utils import detect_storms
from tqdm import tqdm

def count_storms_by_section(data, interval_percent=5, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5):
    """
    Count the number of storms in each section of the data.
    Parameters:
        data: np.ndarray, shape (T, H, W) or (N, C, H, W)
        interval_percent: int, percentage size of each section (e.g., 5 for 5%)
        reflectivity_threshold, area_threshold, dilation_iterations: passed to detect_storms
    Returns:
        List of dicts: [{"section": 0, "start": 0, "end": 10, "storm_count": 20}, ...]
    """
    if data.ndim == 4:
        # Reduce to (T, H, W) by max over channel
        data = np.max(data, axis=1)
    T = data.shape[0]
    section_size = max(1, int(np.ceil(T * interval_percent / 100)))
    sections = []
    n_sections = int(np.ceil(T / section_size))
    for i in tqdm(range(0, T, section_size), desc="Counting storms by section", total=n_sections):
        start = i
        end = min(i + section_size, T)
        storms = detect_storms(data[start:end], reflectivity_threshold, area_threshold, dilation_iterations)
        total_storms = sum(frame["storm_count"] for frame in storms)
        sections.append({
            "section": len(sections),
            "start": start,
            "end": end,
            "storm_count": total_storms
        })
    return sections

def main():
    parser = argparse.ArgumentParser(description="Count storms in sections of radar data using storm_utils.")
    parser.add_argument('--npy_path', type=str, required=True, help='Path to radar data .npy file (T,H,W or N,C,H,W)')
    parser.add_argument('--interval_percent', type=int, default=5, help='Section interval as percent of data length (default: 5)')
    parser.add_argument('--reflectivity_threshold', type=float, default=45, help='Reflectivity threshold (default: 45)')
    parser.add_argument('--area_threshold', type=int, default=15, help='Minimum storm area (default: 15)')
    parser.add_argument('--dilation_iterations', type=int, default=5, help='Dilation iterations (default: 5)')
    parser.add_argument('--out', type=str, default=None, help='Optional output JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        raise FileNotFoundError(f"File not found: {args.npy_path}")
    data = np.load(args.npy_path, mmap_mode='r')
    results = count_storms_by_section(
        data,
        interval_percent=args.interval_percent,
        reflectivity_threshold=args.reflectivity_threshold,
        area_threshold=args.area_threshold,
        dilation_iterations=args.dilation_iterations
    )
    print("Storm counts by section:")
    for section in results:
        print(f"Section {section['section']}: Frames {section['start']}–{section['end']} → {section['storm_count']} storms")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main() 