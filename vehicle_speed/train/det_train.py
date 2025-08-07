import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='YOLO data.yaml path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default='bcs_finetune')
    args = parser.parse_args()

    m = YOLO(args.weights)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)


if __name__ == '__main__':
    main()