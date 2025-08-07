import argparse
import yaml
import cv2
import numpy as np

# Simple placeholder: estimate two vanishing points by line clustering using LSD
# For production, use deep_vp or robust VP estimators.

def detect_lines(gray):
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    return lines if lines is not None else []


def line_to_hessian(line):
    x1, y1, x2, y2 = line
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    v = np.array([a, b, c], dtype=np.float64)
    return v / (np.linalg.norm(v[:2]) + 1e-9)


def intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def estimate_vps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = detect_lines(gray)
    if len(lines) < 2:
        return None, None
    # Naive: sample pairs and cluster intersections by k-means
    pts = []
    Hlines = [line_to_hessian(l.reshape(-1)) for l in lines]
    n = len(Hlines)
    for i in range(min(300, n)):
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        if i1 == i2:
            continue
        p = intersect(Hlines[i1], Hlines[i2])
        if p is None or np.any(np.isnan(p)):
            continue
        if np.all(np.abs(p) < 1e5):
            pts.append(p)
    if len(pts) < 10:
        return None, None
    pts = np.float32(pts)
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(pts, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return centers[0].tolist(), centers[1].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--road_px', type=float, required=True, help='approx pixel distance along lane between two marks')
    parser.add_argument('--road_m', type=float, required=True, help='real distance in meters between those marks')
    parser.add_argument('--out', type=str, default='configs/example_cam.yaml')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    assert img is not None

    vp1, vp2 = estimate_vps(img)
    h, w = img.shape[:2]
    pp = [w/2.0, h/2.0]

    scale = args.road_m / max(args.road_px, 1e-6)

    data = {
        'camera_calibration': {
            'vp1': vp1 if vp1 is not None else [float('nan'), float('nan')],
            'vp2': vp2 if vp2 is not None else [float('nan'), float('nan')],
            'pp': pp,
            'scale': float(scale)
        }
    }
    with open(args.out, 'w') as f:
        yaml.safe_dump(data, f)
    print('Saved', args.out)


if __name__ == '__main__':
    main()