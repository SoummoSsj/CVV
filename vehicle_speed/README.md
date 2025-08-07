# Vehicle Speed Estimation (BrnoCompSpeed-compatible)

This repo provides a practical, real-time vehicle speed estimation pipeline:
- Camera calibration via vanishing points or manual homography
- YOLOv8-based detection + tracking
- World-coordinate projection and per-ID speed estimation
- BrnoCompSpeed evaluation stub

## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset: BrnoCompSpeed
- Request dataset access. Place the extracted folder at `/data/2016-ITS-BrnoCompSpeed`.
- Ground-truth speeds are provided per sequence; this repo reads the pkl/JSON via a lightweight parser in `tools/bcs_eval.py`.

## 3) Calibrate your camera
Two options:
- Auto VP-based: run `tools/calibrate_vp.py` on a short clip; it estimates `vp1`, `vp2`, principal point, and scale (meters per pixel on road plane) using lane direction and vehicle geometry priors.
- Manual homography: click 4+ correspondences between image and known world points using `tools/calibrate_h.py`. Save to a `*.yaml`.

The saved calibration file conforms to the BrnoCompSpeed schema:
```yaml
camera_calibration:
  vp1: [x, y]
  vp2: [x, y]
  pp: [cx, cy]
  scale: 0.01  # meters per px on road plane along lane
```
Or homography:
```yaml
homography:
  H: [ [h11,h12,h13], [h21,h22,h23], [h31,h32,h33] ]
  fps: 30
  scale_m_per_px: 0.01
```

## 4) Train detector (optional)
We default to YOLOv8n. For dataset-specific fine-tuning use `train/det_train.py` with exported frames + labels (YOLO format). For speed estimation, default pretrained weights are fine.

## 5) Run live speed estimation
```bash
python app/live_speed.py --source 0 \
  --weights yolov8n.pt \
  --calib configs/example_cam.yaml \
  --display --save_vid
```
- Shows per-vehicle instantaneous and smoothed speeds in km/h.
- If using homography, set `--use_h`.

## 6) Evaluate on BrnoCompSpeed
```bash
python tools/bcs_eval.py --bcs_root /data/2016-ITS-BrnoCompSpeed \
  --results_dir outputs/bcs_runs/run1
```
Writes result JSONs with `camera_calibration` and `cars:[id, frames, posX, posY]` lists.

## Notes on novelty
This is an engineering integration of known components (YOLOv8, SORT/OC-SORT, VP/homography calibration). It is not research-novel. See ICANN'21 Deep-VP and follow-ups for SOTA calibration + the 2025 efficient estimator for real-time SOTA results.