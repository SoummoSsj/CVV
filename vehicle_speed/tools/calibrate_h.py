import argparse
import yaml
import cv2
import numpy as np

points_img = []
points_world = []

def on_mouse(event, x, y, flags, param):
    global points_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points_img.append([x, y])
        print('Clicked image:', x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='frame image path')
    parser.add_argument('--world_pts', type=str, required=True, help='comma-separated world coords like x1:y1,x2:y2,... in meters')
    parser.add_argument('--out', type=str, default='configs/example_cam.yaml')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--scale_m_per_px', type=float, default=1.0)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    assert img is not None, 'Failed to read image'

    # Parse world points
    for token in args.world_pts.split(','):
        x, y = token.split(':')
        points_world.append([float(x), float(y)])

    print('Click {} image points in order...'.format(len(points_world)))
    cv2.namedWindow('calib')
    cv2.setMouseCallback('calib', on_mouse)

    while True:
        vis = img.copy()
        for p in points_img:
            cv2.circle(vis, tuple(map(int, p)), 5, (0, 0, 255), -1)
        cv2.imshow('calib', vis)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if len(points_img) >= len(points_world):
            break

    cv2.destroyAllWindows()
    assert len(points_img) == len(points_world) and len(points_img) >= 4, 'Need >=4 correspondences'

    src = np.array(points_img, dtype=np.float32)
    dst = np.array(points_world, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
    assert H is not None, 'Homography estimation failed'

    data = {
        'homography': {
            'H': H.tolist(),
            'fps': float(args.fps),
            'scale_m_per_px': float(args.scale_m_per_px)
        }
    }
    with open(args.out, 'w') as f:
        yaml.safe_dump(data, f)
    print('Saved', args.out)


if __name__ == '__main__':
    main()