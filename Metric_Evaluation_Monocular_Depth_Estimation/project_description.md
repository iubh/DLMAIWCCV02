```markdown
# Project Title

Metric Evaluation of Monocular Depth Estimation Using ArUco-Based Plane Geometry

Evaluation of Depth Anything V2 monocular depth predictions against metric ground truth obtained from ArUco marker pose estimation using webcam calibration.

## 1. Project Goal

We evaluate the accuracy of monocular depth prediction by:

- Calibrating a webcam using OpenCV
- Detecting an object using YOLO (yolo26s.pt)
- Detecting an ArUco marker attached to the object
- Estimating the full 6D pose of the marker
- Reconstructing the marker plane in 3D (metric ground truth)
- Backprojecting predicted depth into 3D
- Fitting a predicted plane
- Comparing both planes geometrically

This enables:

- True metric evaluation
- Scale alignment via geometric constraints
- Surface consistency validation

## 2. Updated Folder Structure

```
depthEval_example/
│
├── data/
│   ├── calibration/
│   │   ├── chessboard_images/
│   │   ├── camera_matrix.yaml
│   │   └── dist_coeffs.yaml
│   ├── images/
│   └── markers/
│
├── models/
│   ├── yolo26s.pt
│   └── depth_anything_v2/
│
├── src/
│   ├── calibration/
│   │   ├── capture_chessboard.py
│   │   └── calibrate_camera.py
│   │
│   ├── detection/
│   │   ├── yolo_detector.py
│   │   └── aruco_pose.py
│   │
│   ├── depth/
│   │   ├── depth_anything_wrapper.py
│   │   ├── backprojection.py
│   │   └── plane_alignment.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── visualization/
│   │   └── draw_pose.py
│   │
│   └── main.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
└── README.md
```

## 3. STEP 1 — Webcam Camera Calibration

### 3.1 Why Calibration Is Required

ArUco pose estimation requires:

- Accurate intrinsic matrix (fx, fy, cx, cy)
- Lens distortion coefficients

Without calibration:

- Pose estimation will be wrong
- Metric depth will be unreliable

### 3.2 Chessboard Pattern

Use a standard OpenCV pattern:

- 9x6 inner corners
- Square size: e.g., 25 mm

Print on matte paper and mount flat.

### 3.3 Capture Script (capture_chessboard.py)

Use webcam:

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    found, corners = cv2.findChessboardCorners(gray, (9, 6))
    
    if found:
        cv2.drawChessboardCorners(frame, (9, 6), corners, found)

    cv2.imshow("Calibration", frame)

    if key == 's':
        save_image(frame)
```

Capture:

- 20–40 images
- Different angles
- Different distances
- Entire field of view coverage

Important:

- Tilt board
- Move across image corners
- Vary orientation

### 3.4 Calibration Script (calibrate_camera.py)

Use:

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)
```

Save:

- camera_matrix:
- dist_coeffs:

### 3.5 Calibration Validation

Compute reprojection error:

- < 0.5 pixels → very good
- < 1.0 pixel → acceptable

Print mean reprojection error.

## 4. STEP 2 — ArUco Marker Preparation

Use OpenCV predefined dictionary: `cv2.aruco.DICT_4X4_100`

### 4.1 Marker Printing Instructions

Generate marker:

```python
marker = cv2.aruco.generateImageMarker(dictionary, id, 600)
```

Print with:

- 100% scaling (no auto-fit!)
- High contrast
- Matte paper (no gloss)

Laser printer preferred

Measure marker size precisely using calipers:

Example: Marker side length = 0.048 m

This measurement must be exact for metric accuracy.

### 4.2 Mounting Instructions

- Mount on a flat rigid surface
- Avoid bending
- Ensure marker plane is flat
- Attach directly to object surface

## 5. STEP 3 — ArUco Pose Estimation

Pose estimation:

```python
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners,
    marker_length,
    camera_matrix,
    dist_coeffs
)
```

Output:

- Rotation vector (Rodrigues)
- Translation vector (meters)

Convert to rotation matrix:

```python
R, _ = cv2.Rodrigues(rvec)
```

Marker plane equation:

`n_gt = R[:, 2]`
`d_gt = -n_gt^T * tvec`

This defines the ground truth plane.

## 6. STEP 4 — Depth Anything Integration

Load Depth Anything V2.

Output: Relative depth map `D(u, v)`

## 7. STEP 5 — Backprojection of Predicted Depth

Backproject each pixel inside marker region:

Given intrinsics:

```python
X = (u - cx)/fx * Z
Y = (v - cy)/fy * Z
Z = depth(u, v)
```

This yields predicted 3D points.

## 8. STEP 6 — Plane-Based Alignment (Core Improvement)

Instead of scaling using single Z value:

### 8.1 Extract Marker Pixel Region

Use ArUco corners → create mask.
Select depth pixels inside marker.

### 8.2 Fit Predicted Plane

Use SVD plane fitting:

Given points `P`:

- Compute centroid
- Subtract centroid
- SVD on covariance
- Plane normal = smallest eigenvector

Output: `n_pred, d_pred`

### 8.3 Compute Scale Factor

Because depth is relative:

Find scale `s` minimizing:

`|| n_gt^T (sP_i) + d_gt ||^2`

Closed-form: `s = argmin LS solution`

After scaling: `P_metric = s * P_pred`

## 9. STEP 7 — Evaluation Metrics

Compute:

1. Plane Normal Error:
   - `angle = arccos(n_gt ⋅ n_pred)`

2. Plane Distance Error:
   - `|d_gt - d_pred|`
   
3. RMSE of point-to-plane distance:
   - `RMSE = sqrt(mean((n_gt^T P_pred + d_gt)^2))`

## 10. Main Pipeline

1. Load calibration
2. Capture frame
3. YOLO detection
4. ArUco detection
5. Pose estimation
6. Depth prediction
7. Backproject marker region
8. Fit predicted plane
9. Align scale
10. Compute errors
11. Log + visualize

## 11. Experiments

### Experiment A — Distance Sweep

Move object: `0.3m → 2.5m`

Plot:

- Plane RMSE vs distance
- Scale factor vs distance

### Experiment B — Angle Sweep

Rotate marker: `0° → 60°`

Measure: Normal error increase

### Experiment C — Field-of-View Test

Place marker at:

- Center
- Corners

Check distortion impact.

## 12. Why Plane Alignment Is Better

Single Z comparison:

- Sensitive to noise
- Sensitive to local depth error

Plane fitting:

- Uses many pixels
- Robust to noise
- Evaluates geometry
- Captures surface consistency

This is publication-level evaluation quality.

## 13. Expected Results

Depth Anything provides good relative structure.
Absolute scale varies.
Plane orientation usually accurate.
Bias increases with distance.
Performance degrades at steep angles.

## 14. Master-Level Contributions

This project enables:

- Metric benchmarking of monocular depth
- Evaluation of scale recovery
- Geometry-aware error metrics
- Robust marker-based validation protocol

## 15. README Must Include

Webcam calibration tutorial
Marker printing instructions
Exact marker size recording
Running scripts
Running experiments
Notebook usage
Troubleshooting


## 16. Key Insight

ArUco provides metric geometry from projective constraints.
Depth Anything provides learned relative depth from visual priors.

This project bridges:

Learned depth vs geometric ground truth.
```
