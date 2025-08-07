import argparse
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

from app.live_speed import SimpleTracker, bottom_center, apply_homography, speed_kmh_from_tracks, load_calibration


def process_video(video_path, calib_path, out_json, weights='yolov8n.pt', conf=0.3, use_h=True, classes=(2,3,5,7)):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open {video_path}'
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO(weights)
    tracker = SimpleTracker()

    calib = load_calibration(calib_path)
    H_mat = None
    scale_m_per_px = 1.0
    if use_h and 'homography' in calib and 'H' in calib['homography']:
        H_mat = np.array(calib['homography']['H'], dtype=np.float64)
        scale_m_per_px = float(calib['homography'].get('scale_m_per_px', 1.0))
    elif 'camera_calibration' in calib:
        scale_m_per_px = float(calib['camera_calibration'].get('scale', 1.0))

    id_to_history = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        yres = model.predict(frame, conf=conf, verbose=False)[0]
        dets = []
        classes_np = yres.boxes.cls.cpu().numpy().astype(int) if yres.boxes is not None else []
        boxes_np = yres.boxes.xyxy.cpu().numpy() if yres.boxes is not None else []
        for b, c in zip(boxes_np, classes_np):
            if classes and c not in classes:
                continue
            dets.append(b)
        tracks = tracker.update(dets)

        for tid, t in tracks.items():
            bc = bottom_center(t['bbox'])
            if tid not in id_to_history:
                id_to_history[tid] = []
            if H_mat is not None:
                world_xy = apply_homography(H_mat, bc.reshape(1, 2)).reshape(-1)
            else:
                world_xy = bc
            id_to_history[tid].append({'frame': frame_idx, 'world': world_xy, 'img': bc})
            if len(id_to_history[tid]) > 300:
                id_to_history[tid] = id_to_history[tid][-300:]

    cap.release()

    # Build cars list according to BCS JSON schema
    cars = []
    id_to_speed = speed_kmh_from_tracks(id_to_history, fps=fps, scale_m_per_px=scale_m_per_px)
    for tid, hist in id_to_history.items():
        frames = [h['frame'] for h in hist]
        posX = [float(h['world'][0]) for h in hist]
        posY = [float(h['world'][1]) for h in hist]
        cars.append({
            'id': int(tid),
            'frames': frames,
            'posX': posX,
            'posY': posY,
            'speed_kmh': float(np.median(list(id_to_speed.values())) if tid in id_to_speed else 0.0)
        })

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({
            'camera_calibration': calib.get('camera_calibration', {}),
            'cars': cars
        }, f)
    print('Wrote', out_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--calib', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--use_h', action='store_true')
    args = parser.parse_args()

    process_video(args.video, args.calib, args.out, weights=args.weights, conf=args.conf, use_h=args.use_h)