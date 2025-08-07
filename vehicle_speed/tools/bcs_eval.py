import argparse
import os
import json
import glob
import pickle
import numpy as np

# Minimal stub to aggregate per-car speeds if GT is available

def load_gt_speed_from_pkl(seq_dir):
    # The official dataset includes a .pkl with rich metadata including lines and speeds.
    pkl_files = glob.glob(os.path.join(seq_dir, '*.pkl'))
    if not pkl_files:
        return None
    with open(pkl_files[0], 'rb') as f:
        data = pickle.load(f, encoding='latin1') if hasattr(pickle, 'load') else pickle.load(f)
    # Try common keys (depends on dataset version)
    # Expected structure in original repo caches results; user may adapt this as needed.
    return data


def write_results_json(out_path, camera_calibration, cars):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'camera_calibration': camera_calibration,
            'cars': cars
        }, f)


def simple_eval(pred_speeds, gt_speeds):
    # pred_speeds, gt_speeds: dict car_id -> speed_kmh
    common = set(pred_speeds.keys()) & set(gt_speeds.keys())
    if not common:
        return None
    errs = [abs(pred_speeds[i] - gt_speeds[i]) for i in common]
    return {
        'count': len(common),
        'mae_kmh': float(np.mean(errs)),
        'median_kmh': float(np.median(errs))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bcs_root', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    # Example layout expectation: results_dir/<sequence>.json
    result_files = glob.glob(os.path.join(args.results_dir, '*.json'))
    if not result_files:
        print('No result JSONs found in', args.results_dir)
        return

    # Aggregate simple stats if GT exists per sequence
    all_errs = []
    for rf in result_files:
        seq_name = os.path.splitext(os.path.basename(rf))[0]
        seq_dir = os.path.join(args.bcs_root, seq_name)
        if not os.path.isdir(seq_dir):
            print('Missing sequence dir', seq_dir)
            continue
        gt_data = load_gt_speed_from_pkl(seq_dir)
        if gt_data is None:
            print('No GT .pkl found for', seq_name)
            continue
        # User must adapt this extraction to their GT structure
        gt_speeds = gt_data.get('car_speeds_kmh', {}) if isinstance(gt_data, dict) else {}
        with open(rf, 'r') as f:
            pred = json.load(f)
        pred_speeds = {}
        for car in pred.get('cars', []):
            cid = int(car.get('id', -1))
            sp = float(car.get('speed_kmh', 0.0))
            if cid >= 0:
                pred_speeds[cid] = sp
        stats = simple_eval(pred_speeds, gt_speeds)
        if stats:
            all_errs.append(stats['mae_kmh'])
            print(seq_name, stats)

    if all_errs:
        print('Overall MAE (km/h):', float(np.mean(all_errs)))


if __name__ == '__main__':
    main()