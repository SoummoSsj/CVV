import argparse
import glob
import os
import pickle
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO

from app.live_speed import SimpleTracker, bottom_center


def load_bcs_gt(pkl_path: str) -> Dict:
    with open(pkl_path, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except TypeError:
            data = pickle.load(f)
    return data


def find_spacing_meters(gt: Dict, line_lo: int, line_hi: int) -> float:
    spacings = []
    for car in gt.get('cars', []):
        inter = {x['measurementLineId']: x['videoTime'] for x in car.get('intersections', [])}
        if line_lo in inter and line_hi in inter:
            dt = abs(inter[line_hi] - inter[line_lo])
            if dt <= 0:
                continue
            v_mps = float(car.get('speed', 0.0)) / 3.6
            spacings.append(v_mps * dt)
    if not spacings:
        raise RuntimeError('Cannot derive spacing from GT')
    return float(np.median(spacings))


def line_cross_times(times: List[int], pts_xy: np.ndarray, lines: List[np.ndarray], fps: float) -> Dict[int, float]:
    times_s = {}
    P = np.c_[pts_xy, np.ones((len(pts_xy), 1))]
    for li, l in enumerate(lines):
        s = P @ l  # signed up to scale
        for k in range(len(s) - 1):
            if s[k] == 0:
                times_s[li] = times[k] / fps
                break
            if s[k] * s[k + 1] < 0:
                alpha = -s[k] / (s[k + 1] - s[k] + 1e-9)
                t_cross = (times[k] + alpha) / fps
                times_s[li] = float(t_cross)
                break
    return times_s


def match_track_to_gt(track_cross: Dict[int, float], gt: Dict) -> Tuple[int, float]:
    # returns (carId, error) or (-1, inf)
    best = (-1, 1e18)
    for car in gt.get('cars', []):
        inter = {x['measurementLineId']: x['videoTime'] for x in car.get('intersections', [])}
        common = set(track_cross.keys()) & set(inter.keys())
        if len(common) < 2:
            continue
        err = np.mean([abs(track_cross[i] - inter[i]) for i in common])
        if err < best[1]:
            best = (int(car.get('carId', -1)), float(err))
    return best


class SpeedRegressor(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bcs_session0', type=str, required=True, help='path to session0 root with sequences')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='runs/speed_reg')
    parser.add_argument('--line_lo', type=int, default=2)
    parser.add_argument('--line_hi', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_det = YOLO(args.weights)
    tracker = SimpleTracker()

    X, Y = [], []

    seq_dirs = sorted([d for d in glob.glob(os.path.join(args.bcs_session0, '*')) if os.path.isdir(d)])
    for seq in seq_dirs:
        # load GT
        pkl_files = glob.glob(os.path.join(seq, '*.pkl'))
        if not pkl_files:
            continue
        gt = load_bcs_gt(pkl_files[0])
        fps = float(gt.get('fps', 25.0))
        lines = [np.array(l, dtype=np.float64) for l in gt.get('measurementLines', [])]
        if len(lines) < max(args.line_lo, args.line_hi) + 1:
            continue
        spacing_m = find_spacing_meters(gt, args.line_lo, args.line_hi)

        # open video
        vids = [f for f in glob.glob(os.path.join(seq, '*.mp4'))]
        if not vids:
            continue
        cap = cv2.VideoCapture(vids[0])
        if not cap.isOpened():
            continue
        frame_idx = 0
        tracker.tracks = {}

        # per-track storage: frame indices and image bottom-centers and bbox heights
        history = {}  # tid -> {'frames': [], 'pts': [], 'h': []}

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yres = model_det.predict(frame, conf=0.3, verbose=False)[0]
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
            frame_idx += 1
        cap.release()

        # compute crosses and v_geom per track
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
            v_geom = spacing_m / dt * 3.6  # km/h
            # match to GT to get label
            carId, err = match_track_to_gt(crosses, gt)
            if carId < 0 or err > 0.3:  # loose threshold in seconds
                continue
            # find GT speed
            gt_speed = None
            for car in gt.get('cars', []):
                if int(car.get('carId', -1)) == carId:
                    gt_speed = float(car.get('speed', None))
                    break
            if gt_speed is None:
                continue
            # features
            h_med = float(np.median(rec['h']))
            track_len = len(rec['frames'])
            x_feat = [v_geom, h_med, track_len]
            X.append(x_feat)
            Y.append(gt_speed - v_geom)  # residual

    if not X:
        raise RuntimeError('No training samples collected.')

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

    # normalize features
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    Xn = (X - x_mean) / x_std

    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = SpeedRegressor(in_dim=X.shape[1], hidden=64)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f'Epoch {epoch+1}/{args.epochs} MAE(residual)={total/len(ds):.3f}')

    # save
    torch.save({'state_dict': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std}, os.path.join(args.out_dir, 'speed_reg.pt'))
    meta = {
        'features': ['v_geom', 'bbox_h_median', 'track_len'],
        'line_pair': [args.line_lo, args.line_hi]
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print('Saved model to', os.path.join(args.out_dir, 'speed_reg.pt'))


if __name__ == '__main__':
    main()