import argparse
import time
import yaml
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

# ByteTrack is handled by Ultralytics internally via tracker='bytetrack.yaml'


def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def bottom_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def apply_homography(H, pts_xy):
    pts = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1))], axis=1)
    w = (H @ pts.T).T
    w = w / (w[:, 2:3] + 1e-9)
    return w[:, :2]


def speed_kmh_from_tracks(id_to_history, fps, scale_m_per_px=1.0):
    id_to_speed = {}
    for tid, hist in id_to_history.items():
        if len(hist) < 2:
            continue
        p1 = hist[-2]['world']
        p2 = hist[-1]['world']
        dt = (hist[-1]['frame'] - hist[-2]['frame']) / float(fps)
        if dt <= 0:
            continue
        dist_m = np.linalg.norm((p2 - p1)) * scale_m_per_px
        v_mps = dist_m / dt
        id_to_speed[tid] = v_mps * 3.6
    return id_to_speed


def load_regressor(path):
    if not path:
        return None, None, None
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
    parser.add_argument('--source', type=str, default='0', help='camera index or video path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--calib', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--classes', type=int, nargs='*', default=[2, 3, 5, 7])  # car, motorbike, bus, truck
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--save_vid', action='store_true')
    parser.add_argument('--use_h', action='store_true', help='use homography instead of VP/scale')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--reg', type=str, default='', help='path to trained speed regressor .pt to refine v_geom')
    args = parser.parse_args()

    # Video source
    cap = cv2.VideoCapture(0 if args.source == '0' else args.source)
    assert cap.isOpened(), 'Failed to open source'

    if args.fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        fps = args.fps

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Model with ByteTrack
    model = YOLO(args.weights)

    # Calibration
    calib = load_calibration(args.calib)
    H_mat = None
    scale_m_per_px = 1.0
    if args.use_h and 'homography' in calib and 'H' in calib['homography']:
        H_mat = np.array(calib['homography']['H'], dtype=np.float64)
        scale_m_per_px = float(calib['homography'].get('scale_m_per_px', 1.0))
    elif 'camera_calibration' in calib:
        # For simplicity, treat scale as meters per pixel along road plane
        scale_m_per_px = float(calib['camera_calibration'].get('scale', 1.0))
    else:
        print('Warning: No calibration scale found; defaulting to 1 m/px.')

    # History for speed
    id_to_history = {}
    id_to_bbox_heights = {}

    # Optional regressor
    reg_model, x_mean, x_std = load_regressor(args.reg)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if args.save_vid:
        os.makedirs('outputs', exist_ok=True)
        out = cv2.VideoWriter('outputs/live_speed.mp4', fourcc, fps, (W, Hc))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ByteTrack tracking via Ultralytics
        res = model.track(frame, conf=args.conf, verbose=False, persist=True, tracker='bytetrack.yaml')[0]
        dets = []
        ids = []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            tids = res.boxes.id.cpu().numpy().astype(int)
            for b, c, tid in zip(boxes, clses, tids):
                if args.classes and c not in args.classes:
                    continue
                dets.append(b)
                ids.append(tid)

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # update histories and compute world coords
        for b, tid in zip(dets, ids):
            bc = bottom_center(b)
            if tid not in id_to_history:
                id_to_history[tid] = []
                id_to_bbox_heights[tid] = []
            if H_mat is not None:
                world_xy = apply_homography(H_mat, bc.reshape(1, 2)).reshape(-1)
            else:
                world_xy = bc  # assume near-planar, scaled in pixels
            id_to_history[tid].append({'frame': frame_num, 'world': world_xy})
            id_to_bbox_heights[tid].append(float(b[3] - b[1]))
            # keep recent history
            if len(id_to_history[tid]) > 60:
                id_to_history[tid] = id_to_history[tid][-60:]
                id_to_bbox_heights[tid] = id_to_bbox_heights[tid][-60:]

        id_to_speed_geom = speed_kmh_from_tracks(id_to_history, fps=fps, scale_m_per_px=scale_m_per_px)

        # If regressor is provided, refine speeds
        id_to_speed = {}
        for tid, v_geom in id_to_speed_geom.items():
            if reg_model is None:
                id_to_speed[tid] = v_geom
                continue
            h_med = float(np.median(id_to_bbox_heights.get(tid, [0.0])))
            track_len = len(id_to_history.get(tid, []))
            x = np.array([[v_geom, h_med, track_len]], dtype=np.float32)
            xn = (x - x_mean) / (x_std + 1e-6)
            with torch.no_grad():
                dv = float(reg_model(torch.from_numpy(xn)).cpu().numpy().reshape(-1)[0])
            id_to_speed[tid] = max(0.0, v_geom + dv)

        # Draw
        if res.boxes is not None and res.boxes.id is not None:
            for b, c, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.id.cpu().numpy().astype(int)):
                if args.classes and c not in args.classes:
                    continue
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                v = id_to_speed.get(int(tid), 0.0)
                cv2.putText(frame, f'ID {int(tid)} {v:.1f} km/h', (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if args.display:
            cv2.imshow('live_speed', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if out is not None:
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()