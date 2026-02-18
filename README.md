# DLMAIWCCV02 – Computer Vision Code Collection

This repository is a **course workspace** with small, focused computer-vision examples (OpenCV + webcam + YOLO) and one larger mini-project for **metric evaluation of monocular depth**.

## Repository contents

### 1) OpenCV basics (Notebook)
- `IntroductionOpenCV.ipynb`
  - Image I/O, basic transforms, drawing, filtering, etc.

Run:
```bash
jupyter notebook IntroductionOpenCV.ipynb
```

### 2) Webcam capture utilities
- `webCam.py` – simple webcam stream (press `q` to quit)
- `RecordImagesFromWebCam.py` – capture frames to `frames/` (press `w` to write, `q` to quit)

Run:
```bash
python webCam.py
python RecordImagesFromWebCam.py
```

### 3) YOLO object detection example (Ultralytics)
- `yolo_example.py` – run detection on a single image
- `yolo26s.pt` – model checkpoint, can be obtained from [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

Run:
```bash
python yolo_example.py --image_path bus.jpg --model_size yolo26s.pt --confidence_threshold 0.3
```

### 4) Subproject: Metric Evaluation of Monocular Depth Estimation
Folder:
- `Metric_Evaluation_Monocular_Depth_Estimation/`

Goal:
- Calibrate a webcam (intrinsics + distortion)
- Detect an ArUco marker
- Estimate marker pose and reconstruct its **metric plane**
- Predict monocular depth (pluggable backend; default: MiDaS)
- Backproject depth values inside the marker region, fit a plane, align scale, compute metrics

Start here:
- `Metric_Evaluation_Monocular_Depth_Estimation/README.md`

Minimal quickstart:
```bash
cd Metric_Evaluation_Monocular_Depth_Estimation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (1) capture chessboards, then calibrate
python -m src.calibration.capture_chessboard
python -m src.calibration.calibrate_camera --square_size_m 0.025

# (2) capture one evaluation image
python -m src.main --capture --capture_output data/images/capture.jpg

# (3) run evaluation
python -m src.main --image data/images/capture.jpg --marker_length_m 0.048 --depth_backend midas
```

## Installation

### Root-level examples
Install dependencies for the root-level scripts/notebook:

```bash
pip install -r requirements.txt
```

### Metric evaluation subproject
The subproject has its **own** dependency file:

```bash
pip install -r Metric_Evaluation_Monocular_Depth_Estimation/requirements.txt
```

## Troubleshooting

- **Webcam not accessible**: check permissions / correct camera index.
- **YOLO weights**: ensure `yolo26s.pt` exists in the repo root.
- **ArUco / pose estimation**: see the subproject README (OpenCV wheels differ in `cv2.aruco` capabilities).

## License

Educational use for the DLMAIWCCV02 Computer Vision course.
