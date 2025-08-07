import argparse
import time
import yaml
import cv2
import numpy as np
import os
from ultralytics import YOLO

try:
    from filterpy.kalman import KalmanFilter
    HAVE_FILTERPY = True
except Exception:
    HAVE_FILTERPY = False

# Simple IOU-based tracker fallback
class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections):
        updated_tracks = {}
        used = set()
        # Match existing tracks to detections
        for tid, t in self.tracks.items():
            best_iou = 0
            best_j = -1
            for j, det in enumerate(detections):
                if j in used:
                    continue
                i = self.iou(t['bbox'], det)
                if i > best_iou:
                    best_iou = i
                    best_j = j
            if best_iou >= self.iou_threshold and best_j >= 0:
                updated_tracks[tid] = {'bbox': detections[best_j], 'age': 0}
                used.add(best_j)
            else:
                t['age'] += 1
                if t['age'] <= self.max_age:
                    updated_tracks[tid] = t
        # Create new tracks
        for j, det in enumerate(detections):
            if j in used:
                continue
            updated_tracks[self.next_id] = {'bbox': det, 'age': 0}
            self.next_id += 1
        self.tracks = updated_tracks
        return self.tracks


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

    # Model
    model = YOLO(args.weights)

    # Tracker
    tracker = SimpleTracker()

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if args.save_vid:
        os.makedirs('outputs', exist_ok=True)
        out = cv2.VideoWriter('outputs/live_speed.mp4', fourcc, fps, (W, Hc))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        yres = model.predict(frame, conf=args.conf, verbose=False)[0]
        dets = []
        for b, c in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.cls.cpu().numpy().astype(int)):
            if args.classes and c not in args.classes:
                continue
            dets.append(b)

        tracks = tracker.update(dets)

        # update histories and compute world coords
        for tid, t in tracks.items():
            bc = bottom_center(t['bbox'])
            if tid not in id_to_history:
                id_to_history[tid] = []
            if H_mat is not None:
                world_xy = apply_homography(H_mat, bc.reshape(1, 2)).reshape(-1)
            else:
                world_xy = bc  # assume near-planar, scaled in pixels
            id_to_history[tid].append({'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 'world': world_xy})
            # keep recent history
            if len(id_to_history[tid]) > 30:
                id_to_history[tid] = id_to_history[tid][-30:]

        id_to_speed = speed_kmh_from_tracks(id_to_history, fps=fps, scale_m_per_px=scale_m_per_px)

        # Draw
        for tid, t in tracks.items():
            x1, y1, x2, y2 = map(int, t['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            v = id_to_speed.get(tid, 0.0)
            cv2.putText(frame, f'ID {tid} {v:.1f} km/h', (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

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