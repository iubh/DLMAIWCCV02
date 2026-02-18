# Metric Evaluation of Monocular Depth Estimation (ArUco Plane Geometry)

This project evaluates **monocular depth predictions** against a **metric ground-truth plane** obtained from an **ArUco marker pose** (using calibrated webcam intrinsics).

Core idea:

1. Calibrate webcam (intrinsics + distortion)
2. Detect ArUco marker in an image
3. Estimate marker pose → **metric plane** in camera coordinates
4. Run a monocular depth model → **relative depth**
5. Backproject depths inside the marker area → predicted 3D points
6. Fit plane to predicted points
7. Align scale using the GT plane constraint (least squares)
8. Compute plane-consistency metrics

The depth model is **pluggable**: by default we use **MiDaS** (runs out of the box). A stub for **Depth Anything V2** is included so you can integrate it later.

---

## 1) Installation

From the repository root:

```bash
cd Metric_Evaluation_Monocular_Depth_Estimation

# (recommended) create venv
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### OpenCV ArUco note
If you get an error like `cv2.aruco is not available`, uninstall `opencv-python` and install:

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

If `cv2.aruco` exists but pose helpers like `estimatePoseSingleMarkers` are missing: this project uses
`cv2.solvePnP` internally, so it should still work. Installing `opencv-contrib-python` is still recommended.

---

## 2) Folder structure

```
Metric_Evaluation_Monocular_Depth_Estimation/
├── data/
│   ├── calibration/
│   │   ├── chessboard_images/
│   │   ├── camera_matrix.yaml
│   │   └── dist_coeffs.yaml
│   ├── images/
│   └── markers/
├── models/
│   └── depth_anything_v2/   # optional
├── outputs/
│   ├── visualizations/
│   └── logs/
└── src/
    ├── calibration/
    ├── detection/
    ├── depth/
    ├── evaluation/
    ├── visualization/
    ├── io/
    └── main.py
```

---

## 3) Step-by-step execution

### Step 1 — Webcam calibration

**Goal:** obtain the camera matrix `K` and distortion coefficients `dist` used for correct ArUco pose estimation.

#### 1A) Capture chessboard images

Print a chessboard:
- **9x6 inner corners**
- square size: e.g. **25mm** (0.025m)

Then capture ~20–40 images from different angles and positions:

```bash
python -m src.calibration.capture_chessboard --camera_index 0
```

Controls:
- `s` to save
- `q` to quit

Images will be stored in:
`data/calibration/chessboard_images/`

#### 1B) Run calibration

```bash
python -m src.calibration.calibrate_camera --square_size_m 0.025
```

Outputs:
- `data/calibration/camera_matrix.yaml`
- `data/calibration/dist_coeffs.yaml`

Check the printed reprojection RMSE:
- **< 0.5 px** very good
- **< 1.0 px** acceptable

---

### Step 2 — Print and measure ArUco marker

Dictionary: `DICT_4X4_100`

Optionally generate a marker image with:

```bash
python -m src.detection.generate_marker --id 0 --out data/markers/aruco_0.png
```

Important instructions:
- print at **100% scale** (no “fit to page”)
- high contrast, matte paper
- mount flat (no bending)
- measure side length precisely (calipers)

Default marker size used in CLI examples:

> `--marker_length_m 0.048`

Adjust this if your marker differs.

---

### Step 3 — Capture an evaluation image (optional)

Capture a single image from the webcam and store it in `data/images/`.

```bash
python -m src.main --capture --camera_index 0 --capture_output data/images/capture.jpg
```

Press:
- `SPACE` to save
- `q` to abort

---

### Step 4 — Run metric evaluation on an image

#### 4A) Analyze a specific image

```bash
python -m src.main \
  --image data/images/capture.jpg \
  --marker_length_m 0.048 \
  --depth_backend midas
```

#### 4B) Analyze the latest image in `data/images/`

```bash
python -m src.main --marker_length_m 0.048 --depth_backend midas
```

Outputs:
- prints metrics + scale factor
- saves an overlay image in `outputs/visualizations/`

---

## 4) Metrics explained

All evaluation happens in the **camera coordinate frame**.

1) **Plane normal error (deg)**

Angle between ground-truth normal and predicted normal.

2) **Plane offset error (m)**

Absolute difference between plane offsets `|d_gt - d_pred|`.

3) **RMSE point-to-plane distance (m)**

For backprojected (scaled) points, compute distance to GT plane and report RMSE.

---

## 5) Depth backends (pluggable)

### MiDaS (default, works out-of-the-box)

```bash
python -m src.main --depth_backend midas
```

---

## 6) Jupyter notebook (step-by-step)

An API-driven notebook is provided to explain and execute the pipeline step-by-step:

- `notebooks/step_by_step_depth_eval.ipynb`

Open it with:

```bash
cd Metric_Evaluation_Monocular_Depth_Estimation
jupyter notebook notebooks/step_by_step_depth_eval.ipynb
```

**What it does:**
- loads calibration (`data/calibration/*.yaml`)
- loads an evaluation image from `data/images/`
- detects ArUco + computes the metric ground-truth plane
- runs the MiDaS depth backend
- backprojects points in the marker region
- fits plane + aligns scale + computes metrics
- saves an overlay image to `outputs/visualizations/notebook_overlay.jpg`

**Note:** capture steps are interactive (OpenCV windows) and are therefore kept in the CLI scripts.

### Depth Anything V2 (optional)

`src/depth/depth_anything_v2_backend.py` is a stub.

To integrate DA-v2:
1. Put code/weights under `models/depth_anything_v2/` (or install as dependency)
2. Implement `DepthAnythingV2Backend.predict()` to return a `(H,W)` depth map
3. Run:

```bash
python -m src.main --depth_backend depth_anything_v2
```

---

## 7) Troubleshooting

### Marker not detected
- improve lighting / focus
- ensure dictionary matches (we use `DICT_4X4_100`)
- ensure marker is large enough in the frame
- try without motion blur

### Pose looks wrong / metrics are nonsense
- calibration likely wrong → repeat capture with more varied angles
- check chessboard square size used for calibration
- check marker length measurement in meters

### `cv2.aruco` missing
- install `opencv-contrib-python` (see installation section)

---

## 8) Next steps / experiments

You can extend `src/main.py` to run sweeps described in `project_description.md`:
- distance sweep (0.3m → 2.5m)
- angle sweep (0° → 60°)
- field-of-view placement (center vs corners)
