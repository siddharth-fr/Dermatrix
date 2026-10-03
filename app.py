"""
app.py — Flask web application for SAM 2 lesion segmentation
           with ArUco marker calibration.

Pipeline:
  1. ArUco detection → scale (mm/px) + perspective correction
  2. Preprocess (color constancy, hair removal, glare)
  3. SAM 2 segmentation with box prompt
  4. ABCDE feature extraction with real-world units

Routes:
  GET  /          — upload page with interactive box-drawing canvas
  POST /analyze   — receives image + optional box, returns JSON results
"""

import json
import base64
import traceback

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

from aruco import calibrate, get_calibration_summary, MARKER_PHYSICAL_SIZE_MM
from preprocess import full_preprocess
import segment
from features import extract_all_features
import classifier
import vision_gate
import depth
import time

app = Flask(__name__)

# ---------------------------------------------------------------------------
# SAM 2 model — loaded once at startup
# ---------------------------------------------------------------------------
_sam_loaded = False
_sam_error = None

def _try_load_sam():
    global _sam_loaded, _sam_error
    try:
        from segment import load_model, get_device
        load_model()
        _sam_loaded = True
        print(f"[app] SAM 2 ready on device: {get_device()}")
    except Exception as e:
        _sam_error = str(e)
        print(f"[app] WARNING: SAM 2 failed to load: {e}")
        print("[app] The UI will still load but segmentation won't work.")

_try_load_sam()


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Dermatrix</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="SAM 2 lesion segmentation with ArUco marker calibration for real-world measurements">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #0a0e17; color: #e0e6ed; min-height: 100vh; }

        .header {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-bottom: 1px solid #21262d;
            padding: 20px 0; text-align: center;
        }
        .header h1 { font-size: 1.5rem; color: #58a6ff; font-weight: 700; }
        .header .sub { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
        .disclaimer {
            background: rgba(210, 153, 34, 0.10); color: #d29922;
            text-align: center; padding: 8px 16px; font-size: 0.75rem;
            border-bottom: 1px solid rgba(210, 153, 34, 0.25);
        }

        .main { max-width: 1100px; margin: 24px auto; padding: 0 16px; }

        .card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 10px; padding: 24px; margin-bottom: 20px;
        }
        .card h2 { font-size: 1.1rem; color: #c9d1d9; margin-bottom: 16px; }

        .canvas-wrap {
            position: relative; display: inline-block; max-width: 100%;
            background: #0d1117; border: 1px solid #30363d;
            border-radius: 8px; overflow: hidden; cursor: crosshair;
        }
        .canvas-wrap canvas { display: block; max-width: 100%; }

        .controls { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; align-items: center; }

        .btn {
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white; border: none; padding: 10px 20px;
            border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
            box-shadow: 0 3px 10px rgba(35, 134, 54, 0.2);
        }
        .btn:hover { transform: translateY(-1px); }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .btn-secondary { background: linear-gradient(135deg, #30363d, #3d444d); box-shadow: none; }

        .status { color: #8b949e; font-size: 0.85rem; }
        .status.loading { color: #58a6ff; }
        .status.error { color: #f85149; }
        .status.ok { color: #3fb950; }

        .hint { color: #484f58; font-size: 0.78rem; margin-top: 6px; }

        /* Calibration badge */
        .cal-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
            margin-top: 10px;
        }
        .cal-badge.found { background: rgba(63,185,80,0.12); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
        .cal-badge.none  { background: rgba(139,148,158,0.10); color: #8b949e; border: 1px solid rgba(139,148,158,0.2); }
        .cal-dot { width: 8px; height: 8px; border-radius: 50%; }
        .cal-dot.on  { background: #3fb950; }
        .cal-dot.off { background: #484f58; }

        /* Results grid */
        .results-grid {
            display: grid; grid-template-columns: 1fr 1fr 1fr;
            gap: 16px; margin-top: 16px;
        }
        @media (max-width: 800px) { .results-grid { grid-template-columns: 1fr 1fr; } }
        @media (max-width: 500px) { .results-grid { grid-template-columns: 1fr; } }

        .results-grid img { width: 100%; border-radius: 6px; border: 1px solid #30363d; }
        .results-grid .label {
            font-size: 0.75rem; color: #58a6ff; font-weight: 600;
            margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;
        }

        /* Feature table */
        .feat-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 12px; }
        .feat-table td { padding: 7px 10px; border-bottom: 1px solid #21262d; }
        .feat-table tr:last-child td { border-bottom: none; }
        .feat-table td:first-child { color: #8b949e; }
        .feat-table td:last-child {
            text-align: right; font-weight: 600; color: #e6edf3;
            font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 0.78rem;
        }
        .feat-section {
            color: #58a6ff; font-weight: 700; font-size: 0.75rem;
            padding-top: 10px !important; text-transform: uppercase; letter-spacing: 0.5px;
        }

        input[type=file] { font-size: 14px; color: #8b949e; cursor: pointer; }

        /* Calibration details table */
        .cal-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 10px; }
        .cal-table td { padding: 5px 8px; border-bottom: 1px solid #21262d; }
        .cal-table td:first-child { color: #8b949e; }
        .cal-table td:last-child { text-align: right; color: #c9d1d9; font-weight: 500; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Dermatrix</h1>
        <div class="sub">SAM 2 Segmentation · ArUco Scale Calibration · ABCDE Features · Perspective Correction</div>
    </div>
    <div class="disclaimer">
        ⚠️ Research / measurement tool only — NOT a medical diagnosis. Do not use for clinical decisions.
    </div>

    <div class="main">
        <!-- Upload Card -->
        <div class="card">
            <h2>1. Upload & Draw Box</h2>
            <input type="file" id="fileInput" accept="image/*">
            <div class="hint">
                Upload a lesion photo with an ArUco marker (13.5 mm) visible for automatic scale calibration.
                Then draw a rectangle around the lesion, or click on it.
            </div>

            <div class="canvas-wrap" id="canvasWrap" style="display:none; margin-top: 14px;">
                <canvas id="imgCanvas"></canvas>
            </div>

            <div class="controls">
                <button class="btn" id="btnAnalyze" disabled>Analyze with SAM 2</button>
                <button class="btn btn-secondary" id="btnClear" disabled>Clear Box</button>
                <button class="btn btn-secondary" id="btnAuto" disabled>Auto-detect Box</button>
                <span class="status" id="status"></span>
            </div>
        </div>

        <!-- Results Card (initially hidden) -->
        <div class="card" id="resultsCard" style="display:none;">
            <!-- ArUco calibration info -->
            <div id="calCard"></div>

            <h2 style="margin-top: 16px;">2. Results</h2>
            <div class="results-grid">
                <div>
                    <div class="label">Original</div>
                    <img id="imgOriginal">
                </div>
                <div>
                    <div class="label">Segmentation Overlay</div>
                    <img id="imgOverlay">
                </div>
                <div>
                    <div class="label">Cropped Lesion</div>
                    <img id="imgCropped">
                </div>
            </div>

            <h2 style="margin-top: 20px;">3. ABCDE Features</h2>
            <table class="feat-table" id="featTable"></table>
        </div>
    </div>

    <script>
    const fileInput = document.getElementById('fileInput');
    const canvas = document.getElementById('imgCanvas');
    const ctx = canvas.getContext('2d');
    const canvasWrap = document.getElementById('canvasWrap');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnClear = document.getElementById('btnClear');
    const btnAuto = document.getElementById('btnAuto');
    const statusEl = document.getElementById('status');

    let originalImg = null;
    let imgFile = null;
    let box = null;
    let drawing = false;
    let startX, startY;
    let displayScale = 1;

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        imgFile = file;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => {
                originalImg = img;
                const maxW = Math.min(700, canvasWrap.parentElement.clientWidth - 50);
                displayScale = Math.min(1, maxW / img.width);
                canvas.width = Math.round(img.width * displayScale);
                canvas.height = Math.round(img.height * displayScale);
                drawImage();
                canvasWrap.style.display = 'inline-block';
                btnAnalyze.disabled = false;
                btnClear.disabled = false;
                btnAuto.disabled = false;
                box = null;
                statusEl.textContent = 'Draw a box around the lesion, or click Analyze for auto-detection.';
                statusEl.className = 'status';
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });

    function drawImage() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
        if (box) {
            ctx.strokeStyle = '#f85149';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(
                box.x1 * displayScale, box.y1 * displayScale,
                (box.x2 - box.x1) * displayScale, (box.y2 - box.y1) * displayScale
            );
            ctx.setLineDash([]);
        }
    }

    // Mouse drawing
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        startX = (e.clientX - rect.left) / displayScale;
        startY = (e.clientY - rect.top) / displayScale;
        drawing = true;
    });
    canvas.addEventListener('mousemove', (e) => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const cx = (e.clientX - rect.left) / displayScale;
        const cy = (e.clientY - rect.top) / displayScale;
        box = {
            x1: Math.round(Math.min(startX, cx)), y1: Math.round(Math.min(startY, cy)),
            x2: Math.round(Math.max(startX, cx)), y2: Math.round(Math.max(startY, cy))
        };
        drawImage();
    });
    canvas.addEventListener('mouseup', (e) => {
        if (!drawing) return;
        drawing = false;
        const rect = canvas.getBoundingClientRect();
        const ex = (e.clientX - rect.left) / displayScale;
        const ey = (e.clientY - rect.top) / displayScale;
        if (Math.abs(ex - startX) < 5 && Math.abs(ey - startY) < 5) {
            const w = originalImg.width * 0.2, h = originalImg.height * 0.2;
            box = {
                x1: Math.round(Math.max(0, startX - w)),
                y1: Math.round(Math.max(0, startY - h)),
                x2: Math.round(Math.min(originalImg.width, startX + w)),
                y2: Math.round(Math.min(originalImg.height, startY + h))
            };
        }
        drawImage();
        if (box) {
            statusEl.textContent = `Box set: (${box.x1}, ${box.y1}) to (${box.x2}, ${box.y2})`;
            statusEl.className = 'status ok';
        }
    });

    // Touch support
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const t = e.touches[0], r = canvas.getBoundingClientRect();
        startX = (t.clientX - r.left) / displayScale;
        startY = (t.clientY - r.top) / displayScale;
        drawing = true;
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!drawing) return;
        const t = e.touches[0], r = canvas.getBoundingClientRect();
        const cx = (t.clientX - r.left) / displayScale;
        const cy = (t.clientY - r.top) / displayScale;
        box = {
            x1: Math.round(Math.min(startX, cx)), y1: Math.round(Math.min(startY, cy)),
            x2: Math.round(Math.max(startX, cx)), y2: Math.round(Math.max(startY, cy))
        };
        drawImage();
    });
    canvas.addEventListener('touchend', () => {
        drawing = false;
        if (box) {
            statusEl.textContent = `Box set: (${box.x1}, ${box.y1}) to (${box.x2}, ${box.y2})`;
            statusEl.className = 'status ok';
        }
        drawImage();
    });

    btnClear.addEventListener('click', () => {
        box = null; drawImage();
        statusEl.textContent = 'Box cleared.'; statusEl.className = 'status';
    });

    btnAuto.addEventListener('click', () => { box = null; drawImage(); doAnalyze(); });
    btnAnalyze.addEventListener('click', () => doAnalyze());

    function doAnalyze() {
        if (!imgFile) return;
        btnAnalyze.disabled = true;
        statusEl.textContent = 'Running ArUco + SAM 2 pipeline...';
        statusEl.className = 'status loading';

        const fd = new FormData();
        fd.append('image', imgFile);
        if (box) fd.append('box', JSON.stringify([box.x1, box.y1, box.x2, box.y2]));

        fetch('/analyze', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                btnAnalyze.disabled = false;
                if (data.error) {
                    statusEl.textContent = 'Error: ' + data.error;
                    statusEl.className = 'status error';
                    return;
                }
                statusEl.textContent = `Done - SAM score: ${data.sam_score.toFixed(3)}`;
                statusEl.className = 'status ok';

                document.getElementById('resultsCard').style.display = 'block';
                document.getElementById('imgOriginal').src = 'data:image/jpeg;base64,' + data.original_b64;
                document.getElementById('imgOverlay').src = 'data:image/jpeg;base64,' + data.overlay_b64;
                document.getElementById('imgCropped').src = 'data:image/png;base64,' + data.cropped_b64;

                // Calibration card
                const calCard = document.getElementById('calCard');
                const cal = data.calibration;
                if (cal.aruco_status && cal.aruco_status.includes('detected')) {
                    let html = '<div class="cal-badge found"><span class="cal-dot on"></span> ArUco Calibrated</div>';
                    html += '<table class="cal-table">';
                    for (const [k, v] of Object.entries(cal)) {
                        html += `<tr><td>${k}</td><td>${v}</td></tr>`;
                    }
                    html += '</table>';
                    calCard.innerHTML = html;
                } else {
                    calCard.innerHTML = '<div class="cal-badge none"><span class="cal-dot off"></span> No ArUco marker — pixel-only measurements</div>';
                }

                // Feature table
                const table = document.getElementById('featTable');
                table.innerHTML = '';
                const sections = {
                    'area_px': 'GEOMETRY',
                    'asymmetry_axis1': 'ASYMMETRY',
                    'border_irregularity': 'BORDER',
                    'mean_rgb_lesion': 'COLOR',
                };
                for (const [key, val] of Object.entries(data.features)) {
                    if (sections[key]) {
                        const sr = document.createElement('tr');
                        sr.innerHTML = `<td class="feat-section" colspan="2">${sections[key]}</td>`;
                        table.appendChild(sr);
                    }
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${key}</td><td>${val}</td>`;
                    table.appendChild(row);
                }

                if (data.box_used) {
                    box = { x1: data.box_used[0], y1: data.box_used[1],
                            x2: data.box_used[2], y2: data.box_used[3] };
                    drawImage();
                }
            })
            .catch(err => {
                btnAnalyze.disabled = false;
                statusEl.textContent = 'Network error: ' + err.message;
                statusEl.className = 'status error';
            });
    }
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full pipeline:
      1. ArUco calibration (scale + perspective correction)
      2. Preprocess
      3. SAM 2 segmentation
      4. Feature extraction with real-world units
    """
    if not _sam_loaded:
        return jsonify({"error": f"SAM 2 model not available: {_sam_error}"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        from segment import segment as sam_segment

        # Decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Could not decode image."}), 400

        # Parse optional box
        box = None
        box_str = request.form.get('box')
        if box_str:
            try:
                box = np.array(json.loads(box_str), dtype=np.float32)
            except Exception:
                pass

        # ── STAGE 1: ArUco calibration ──
        cal = calibrate(img_bgr)
        cal_summary = get_calibration_summary(cal)
        mm_per_px = cal.get("corrected_mm_per_px") or cal.get("mm_per_px")

        # Use the perspective-corrected image if markers were found and tilt is significant
        if cal["found"] and cal["corrected_image"] is not None:
            work_img = cal["corrected_image"]
            H = cal["homography"]
            print(f"[app] ArUco: marker ID {cal['best_marker']['id']}, "
                  f"scale={mm_per_px:.4f} mm/px, "
                  f"tilt={cal['geometry']['tilt_angle_deg']}deg")

            # Transform the box to corrected coordinates if user drew one
            if box is not None and H is not None:
                pts = np.array([[box[0], box[1]], [box[2], box[1]],
                                [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
                ones = np.ones((4, 1))
                homog = np.hstack([pts, ones])
                transformed = (H @ homog.T).T
                w = transformed[:, 2:3]; w[w == 0] = 1e-8
                xy = transformed[:, :2] / w
                box = np.array([
                    xy[:, 0].min(), xy[:, 1].min(),
                    xy[:, 0].max(), xy[:, 1].max()
                ], dtype=np.float32)
        else:
            work_img = img_bgr
            H = None
            print("[app] ArUco: no marker found, using pixel-only measurements")

        # ── STAGE 2: Preprocess ──
        preprocessed = full_preprocess(work_img)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        # ── STAGE 3: SAM 2 segmentation ──
        result = sam_segment(rgb, box=box)
        mask = result["mask"]
        contour = result["contour"]
        used_box = result["box"]
        sam_score = result["score"]

        # ── STAGE 4: Feature extraction ──
        features = extract_all_features(work_img, contour, mask, mm_per_px)

        # Add a scale source indicator
        if mm_per_px:
            features["scale_source"] = f"ArUco ({MARKER_PHYSICAL_SIZE_MM}mm marker)"
            features["mm_per_px"] = float(round(mm_per_px, 5))
        else:
            features["scale_source"] = "None (pixel only)"

        # Pop debug contours to prevent JSON serialization errors and use for visualization
        debug_contours = features.pop("debug_contours", None)

        # ── Visualization ──
        # Draw detected marker outlines on original
        if cal["found"]:
            for m in cal["markers"]:
                pts_int = m["corners"].astype(np.int32).reshape((-1, 1, 2))
                # Draw the quadrilateral
                cv2.polylines(img_bgr, [pts_int], True, (0, 255, 255), 3)
                # Draw a bounding box around it
                x, y, w_box, h_box = cv2.boundingRect(pts_int)
                cv2.rectangle(img_bgr, (x, y), (x + w_box, y + h_box), (0, 200, 255), 2)
                cv2.putText(img_bgr, f"Day 1, ID: {m['id']}", (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w_box / 100.0), (0, 200, 255), 2)

        # Original (show the uncorrected image)
        orig_display = _resize_for_web(img_bgr)
        _, orig_buf = cv2.imencode('.jpg', orig_display, [cv2.IMWRITE_JPEG_QUALITY, 88])

        x1, y1, x2, y2 = used_box

        # Step 1: Bounding Box
        step1 = work_img.copy()
        cv2.rectangle(step1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        _, step1_buf = cv2.imencode('.jpg', _resize_for_web(step1), [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Step 2: Bounding Box + SAM Mask
        step2 = step1.copy()
        colored = np.zeros_like(work_img)
        colored[mask > 0] = [255, 140, 0]
        cv2.addWeighted(colored, 0.35, step2, 0.65, 0, step2)
        _, step2_buf = cv2.imencode('.jpg', _resize_for_web(step2), [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Step 3: Box + Mask + Raw Contour
        step3 = step2.copy()
        if debug_contours:
            cv2.drawContours(step3, [debug_contours["raw"]], -1, (0, 0, 255), 1)
        else:
            cv2.drawContours(step3, [contour], -1, (0, 0, 255), 1)
        _, step3_buf = cv2.imencode('.jpg', _resize_for_web(step3), [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Final Overlay (Adds Green Smooth Contour)
        overlay = step3.copy()
        if debug_contours:
            cv2.drawContours(overlay, [debug_contours["light_smooth"]], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        _, overlay_buf = cv2.imencode('.jpg', _resize_for_web(overlay), [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Cropped lesion (background blacked out, tight crop)
        cropped = work_img.copy()
        cropped[mask == 0] = 0
        
        # Save a copy for AI classification before drawing artificial green contours
        ai_input = cropped.copy()
        
        # Draw the perimeter (contour) on the cropped lesion for the UI
        if debug_contours:
            cv2.drawContours(cropped, [debug_contours["light_smooth"]], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(cropped, [contour], -1, (0, 255, 0), 2)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            pad = max(10, int(max(bw, bh) * 0.05))
            cropped = cropped[
                max(0, by - pad):min(work_img.shape[0], by + bh + pad),
                max(0, bx - pad):min(work_img.shape[1], bx + bw + pad)
            ]
            ai_input = ai_input[
                max(0, by - pad):min(work_img.shape[0], by + bh + pad),
                max(0, bx - pad):min(work_img.shape[1], bx + bw + pad)
            ]
            
        # ── AI Classification ──
        # Predict on the isolated, tightly cropped lesion (without green line)
        ai_result = classifier.classify_lesion(ai_input)
        
        # ── Depth Estimation (Wider Crop using used_box) ──
        # We use the full used_box to get a lot of skin in a rect shape.
        x1_b, y1_b, x2_b, y2_b = [int(v) for v in used_box]
        # Ensure box is within image bounds
        x1_b = max(0, x1_b)
        y1_b = max(0, y1_b)
        x2_b = min(work_img.shape[1], x2_b)
        y2_b = min(work_img.shape[0], y2_b)
        
        # Raw image for depth estimation
        depth_img_raw = work_img[y1_b:y2_b, x1_b:x2_b]
        
        # Raw image (without hitbox and contour) for the 3D texture
        depth_img_texture = work_img[y1_b:y2_b, x1_b:x2_b]
        
        # Generate depth map (no mask used to allow depth estimation on skin)
        depth_map = depth.estimate_depth(depth_img_raw)
        
        depth_b64 = None
        depth_texture_b64 = None
        if depth_map is not None:
            _, depth_buf = cv2.imencode('.png', depth_map)
            depth_b64 = base64.b64encode(depth_buf).decode()
            
            _, depth_tex_buf = cv2.imencode('.png', depth_img_texture)
            depth_texture_b64 = base64.b64encode(depth_tex_buf).decode()
        
        _, crop_buf = cv2.imencode('.png', cropped)

        return jsonify({
            "original_b64": base64.b64encode(orig_buf).decode(),
            "step1_b64": base64.b64encode(step1_buf).decode(),
            "step2_b64": base64.b64encode(step2_buf).decode(),
            "step3_b64": base64.b64encode(step3_buf).decode(),
            "overlay_b64": base64.b64encode(overlay_buf).decode(),
            "cropped_b64": base64.b64encode(crop_buf).decode(),
            "depth_b64": depth_b64,
            "depth_texture_b64": depth_texture_b64,
            "features": features,
            "calibration": cal_summary,
            "box_used": used_box.tolist(),
            "sam_score": float(sam_score),
            "ai_classification": ai_result
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline error: {e}"}), 500


def _resize_for_web(img, max_dim=800):
    """Resize for web transfer."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img.copy()
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect_aruco', methods=['POST'])
def detect_aruco():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image"}), 400
            
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        from aruco import detect_markers, MARKER_PHYSICAL_SIZE_MM
        markers = detect_markers(img_bgr)
        
        # Approximate camera intrinsic matrix
        h, w = img_bgr.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        ml = MARKER_PHYSICAL_SIZE_MM
        obj_pts = np.array([
            [-ml/2,  ml/2, 0],
            [ ml/2,  ml/2, 0],
            [ ml/2, -ml/2, 0],
            [-ml/2, -ml/2, 0]
        ], dtype=np.float32)
        
        axis = np.array([
            [0, 0, 0],       # Origin
            [ml, 0, 0],      # X (Red)
            [0, ml, 0],      # Y (Green)
            [0, 0, ml]       # Z (Blue)
        ], dtype=np.float32)
        
        results = []
        for m in markers:
            corners = m["corners"]
            success, rvec, tvec = cv2.solvePnP(obj_pts, corners, camera_matrix, dist_coeffs)
            axes_2d = None
            if success:
                img_pts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                axes_2d = img_pts.reshape(-1, 2).tolist()
                
            results.append({
                "id": int(m["id"]),
                "corners": corners.tolist(),
                "axes": axes_2d
            })
            
        return jsonify({"markers": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Gemini Vision Gate API
# ---------------------------------------------------------------------------
@app.route('/api/settings/gemini', methods=['GET', 'POST'])
def gemini_settings():
    if request.method == 'GET':
        masked_key = ""
        if vision_gate.config["api_key"]:
            masked_key = f"sk-...{vision_gate.config['api_key'][-4:]}" if len(vision_gate.config["api_key"]) >= 4 else "sk-***"
            
        return jsonify({
            "model": vision_gate.config["model"],
            "api_key_configured": bool(vision_gate.config["api_key"]),
            "masked_key": masked_key
        })
        
    data = request.json
    action = data.get("action")
    
    if action == "save":
        api_key = data.get("api_key")
        model = data.get("model")
        vision_gate.save_config(api_key=api_key, model=model)
        return jsonify({"success": True})
        
    elif action == "clear":
        vision_gate.clear_config()
        return jsonify({"success": True})
        
    elif action == "test":
        test_key = data.get("api_key") or vision_gate.config["api_key"]
        test_model = data.get("model") or vision_gate.config["model"]
        
        if not test_key:
            return jsonify({"status": "CONNECTION FAILED", "error": "No API key provided"})
            
        result = vision_gate.test_connection(test_key, test_model)
        return jsonify(result)

    return jsonify({"error": "Unknown action"}), 400

@app.route('/api/vision-gate', methods=['POST'])
def vision_gate_check():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
        
    try:
        file_bytes = file.read()
        
        # Determine mime type from filename
        mime_type = "image/jpeg"
        if file.filename.lower().endswith(".png"):
            mime_type = "image/png"
        elif file.filename.lower().endswith(".webp"):
            mime_type = "image/webp"
            
        start_time = time.time()
        gate_result = vision_gate.analyze_image_validity(file_bytes, mime_type)
        latency_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            "decision": gate_result.get("decision", "REVIEW"),
            "gate_result": gate_result,
            "model": vision_gate.config["model"],
            "latency_ms": latency_ms
        })
    except Exception as e:
        error_msg = str(e)
        if "AI_GATE_UNAVAILABLE" in error_msg:
            return jsonify({
                "decision": "AI_GATE_UNAVAILABLE",
                "error": error_msg
            }), 503
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
