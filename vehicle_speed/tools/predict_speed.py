import argparse
import json
import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.live_speed import SimpleTracker, bottom_center
from train.speed_reg_train import line_cross_times, load_bcs_gt, find_spacing_meters


def load_regressor(path):
    ckpt = torch.load(path, map_location='cpu')
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['x_mean'], ckpt['x_std']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--bcs_gt_pkl', type=str, required=True)
    parser.add_argument('--reg', type=str, required=True, help='path to runs/speed_reg/speed_reg.pt')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--line_lo', type=int, default=2)
    parser.add_argument('--line_hi', type=int, default=0)
    args = parser.parse_args()

    model_det = YOLO(args.weights)
    tracker = SimpleTracker()
    model_reg, x_mean, x_std = load_regressor(args.reg)

    gt = load_bcs_gt(args.bcs_gt_pkl)
    fps = float(gt.get('fps', 25.0))
    lines = [np.array(l, dtype=np.float64) for l in gt.get('measurementLines', [])]
    spacing_m = find_spacing_meters(gt, args.line_lo, args.line_hi)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    history = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        yres = model_det.predict(frame, conf=args.conf, verbose=False)[0]
        dets = []
        for b in (yres.boxes.xyxy.cpu().numpy() if yres.boxes is not None else []):
            dets.append(b)
        tracks = tracker.update(dets)
        for tid, t in tracks.items():
            bc = bottom_center(t['bbox'])
            x1, y1, x2, y2 = t['bbox']
            h = y2 - y1
            if tid not in history:
                history[tid] = {'frames': [], 'pts': [], 'h': []}
            history[tid]['frames'].append(frame_idx)
            history[tid]['pts'].append(bc.tolist())
            history[tid]['h'].append(float(h))
    cap.release()

    cars = []
    for tid, rec in history.items():
        if len(rec['frames']) < 5:
            continue
        pts = np.array(rec['pts'], dtype=np.float64)
        crosses = line_cross_times(rec['frames'], pts, lines, fps)
        if args.line_lo not in crosses or args.line_hi not in crosses:
            continue
        dt = abs(crosses[args.line_hi] - crosses[args.line_lo])
        if dt <= 0:
            continue
        v_geom = spacing_m / dt * 3.6
        h_med = float(np.median(rec['h']))
        track_len = len(rec['frames'])
        x = np.array([[v_geom, h_med, track_len]], dtype=np.float32)
        xn = (x - x_mean) / (x_std + 1e-6)
        with torch.no_grad():
            dv = float(model_reg(torch.from_numpy(xn)).cpu().numpy().reshape(-1)[0])
        v_hat = max(0.0, v_geom + dv)

        cars.append({
            'id': int(tid),
            'frames': rec['frames'],
            'posX': [float(p[0]) for p in rec['pts']],
            'posY': [float(p[1]) for p in rec['pts']],
            'speed_kmh': float(v_hat),
            'intersections': [{'measurementLineId': int(k), 'videoTime': float(v)} for k, v in sorted(crosses.items())]
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'camera_calibration': {},
            'cars': cars
        }, f)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()