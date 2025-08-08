# Kaggle Speed Estimation (BrnoCompSpeed)

This folder contains a Kaggle-ready notebook to train a from-scratch speed residual model (no pretrained speed weights), using YOLOv8 + ByteTrack for detection/tracking, and the BrnoCompSpeed ground-truth for supervision.

## Usage on Kaggle
1) Upload/attach the BrnoCompSpeed dataset to your Kaggle session. A common mount is `/kaggle/input/brnocompspeed/2016-ITS-BrnoCompSpeed`.
2) Open `kaggle_speed.ipynb`.
3) Set the dataset paths in the first cell to your Kaggle input locations.
4) Run all cells to:
   - Install dependencies (Ultralytics, OpenCV, PyTorch comes preinstalled)
   - Train the residual speed model on `session0`
   - Run prediction on a chosen sequence and export `SEQ.json`

Outputs are saved under `/kaggle/working/`.