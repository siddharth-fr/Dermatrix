# Dermatrix

Skin lesion segmentation and ABCDE analysis powered by **Segment Anything Model 2 (SAM 2)** with **ArUco marker calibration** for real-world measurements.

> ⚠️ **Research / measurement tool only** — NOT a medical diagnosis tool.

## Architecture

```
app.py           — Flask web server, UI, /analyze API endpoint
aruco.py         — ArUco marker detection, scale calibration, perspective correction
preprocess.py    — Color constancy (Minkowski p=6), hair removal, glare inpainting
segment.py       — SAM 2 model loading, box estimation, mask validation
features.py      — ABCDE feature extraction (Asymmetry, Border, Color, Diameter)
tracking.py      — Longitudinal comparison (ORB/RANSAC alignment)
evaluate.py      — Batch evaluation on ISIC/PH2 datasets
test_features.py — Unit tests for feature computation
```

## Pipeline

```
Image → ArUco Detect → Perspective Correct → Preprocess → SAM 2 Segment → Features → UI
                ↓                                                              ↓
         mm/px scale                                              Real-world mm measurements
         tilt angle
```

## ArUco Marker

The system uses ArUco markers with a **known physical size of 13.5 mm (1.35 cm)** per side.
When a marker is detected in the image:
- **Scale (mm/px)** is computed from the marker's pixel side length
- **Perspective correction** dewarps the image using a homography so the marker becomes a perfect square
- **Tilt angle** is estimated from the foreshortening of the marker sides
- All measurements (area, diameter, perimeter) are reported in **real millimeters**

If no marker is found, measurements are reported in pixels only.

## Install

```bash
pip install -r requirements.txt
```

### SAM 2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

On Windows without CUDA build tools, use:
```bash
$env:SAM2_BUILD_CUDA="0"; pip install git+https://github.com/facebookresearch/sam2.git
```

## Usage

```bash
python app.py
# Open http://127.0.0.1:5000
```

1. Upload an image (with ArUco marker visible for calibration)
2. Draw a rectangle around the lesion (or click for auto-sized box)
3. Click **Analyze with SAM 2**
4. View segmentation overlay, cropped lesion, calibration data, and ABCDE features

### Run unit tests

```bash
python test_features.py
```

### Batch evaluation

```bash
python evaluate.py --images ./data/images --masks ./data/masks --out ./eval_results
```
