"""
features.py — ABCDE-style lesion feature extraction with multi-scale contour smoothing.

Computes:
  A — Asymmetry (PCA-aligned flip along both axes)
  B — Border irregularity (radial distance profile, multi-scale noise decoupling)
  C — Color (k-means clustering, dominant colors, lesion vs skin comparison)
  D — Diameter / area / scale-aware geometric features
  E — Evolution (placeholder — needs two images, handled in tracking.py)
"""

import cv2
import numpy as np
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Contour Smoothing (Decoupling noise from genuine structure)
# ---------------------------------------------------------------------------
def smooth_contour(contour: np.ndarray, num_points: int = 300, smooth_factor: float = 0.05) -> np.ndarray:
    """
    Uniformly resample contour by arc length and apply scale-aware moving average.
    
    smooth_factor is a fraction of the total contour length (e.g., 0.05 = 5% of perimeter).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return contour
        
    # Ensure closed contour for proper interpolation
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack((pts, pts[0]))
    
    # Parameterize by arc length
    dp = np.diff(pts, axis=0)
    arc = np.zeros(len(pts))
    arc[1:] = np.cumsum(np.linalg.norm(dp, axis=1))
    
    total_len = arc[-1]
    if total_len == 0:
        return contour
        
    # Resample uniformly
    arc_uniform = np.linspace(0, total_len, num_points)
    x_unif = np.interp(arc_uniform, arc, pts[:, 0])
    y_unif = np.interp(arc_uniform, arc, pts[:, 1])
    
    window_length = int(num_points * smooth_factor)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(3, window_length)
    
    if window_length < num_points:
        window = np.ones(window_length) / window_length
        pad_size = window_length
        
        # Wrap around for closed contour
        x_pad = np.concatenate((x_unif[-pad_size:], x_unif, x_unif[:pad_size]))
        y_pad = np.concatenate((y_unif[-pad_size:], y_unif, y_unif[:pad_size]))
        
        x_smooth = np.convolve(x_pad, window, mode='same')[pad_size:-pad_size]
        y_smooth = np.convolve(y_pad, window, mode='same')[pad_size:-pad_size]
    else:
        x_smooth, y_smooth = x_unif, y_unif
        
    smoothed = np.column_stack((x_smooth, y_smooth)).reshape(-1, 1, 2).astype(np.int32)
    return smoothed


# ---------------------------------------------------------------------------
# A - Geometry
# ---------------------------------------------------------------------------
def compute_geometry(contour: np.ndarray, mm_per_px: float = None) -> OrderedDict:
    """
    Scale-aware geometric features calculated on the smoothed contour.
    """
    feat = OrderedDict()
    if mm_per_px is not None:
        mm_per_px = float(mm_per_px)

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    feat["area_px"] = int(area)
    if mm_per_px:
        feat["area_mm2"] = round(area * mm_per_px ** 2, 2)
        feat["area_mm2_note"] = "area (volume proxy)"
    else:
        feat["area_mm2"] = "N/A (no scale)"

    feat["perimeter_px"] = round(perimeter, 1)
    if mm_per_px:
        feat["perimeter_mm"] = round(perimeter * mm_per_px, 2)
    else:
        feat["perimeter_mm"] = "N/A (no scale)"

    # Circularity
    if perimeter > 0:
        feat["circularity"] = round(4 * np.pi * area / (perimeter ** 2), 4)
    else:
        feat["circularity"] = 0.0

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    feat["solidity"] = round(area / (hull_area + 1e-8), 4)

    # Roundness & Axes
    if len(contour) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(contour)
        major = max(MA, ma)
        minor = min(MA, ma)
        feat["roundness"] = round(4 * area / (np.pi * major ** 2 + 1e-8), 4)
        feat["eccentricity"] = round(np.sqrt(1 - (minor**2 / (major**2 + 1e-8))), 4)
    else:
        feat["roundness"] = 0.0
        feat["eccentricity"] = 0.0

    # Aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    feat["aspect_ratio"] = round(max(w, h) / (min(w, h) + 1e-8), 3)

    # Diameter
    feat["diameter_px"] = round(float(np.sqrt(4 * area / np.pi)), 1)
    if mm_per_px:
        feat["diameter_mm"] = round(feat["diameter_px"] * mm_per_px, 2)
    else:
        feat["diameter_mm"] = "N/A (no scale)"

    return feat


def compute_asymmetry(mask: np.ndarray) -> OrderedDict:
    """
    Asymmetry via PCA alignment + flip on the cleaned mask.
    """
    feat = OrderedDict()

    pts = np.column_stack(np.where(mask > 0)).astype(np.float64)  # (N, 2) — row, col
    if len(pts) < 10:
        feat["asymmetry_axis1"] = 0.0
        feat["asymmetry_axis2"] = 0.0
        feat["asymmetry_mean"] = 0.0
        return feat

    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    cov = np.cov(pts_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    h, w = mask.shape
    angle = np.degrees(np.arctan2(eigvecs[0, 0], eigvecs[1, 0]))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

    rows = np.any(rotated, axis=1)
    cols = np.any(rotated, axis=0)
    if not np.any(rows) or not np.any(cols):
        feat["asymmetry_axis1"] = 0.0
        feat["asymmetry_axis2"] = 0.0
        feat["asymmetry_mean"] = 0.0
        return feat

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = rotated[rmin:rmax + 1, cmin:cmax + 1]

    lesion_area = np.count_nonzero(cropped)
    if lesion_area == 0:
        feat["asymmetry_axis1"] = 0.0
        feat["asymmetry_axis2"] = 0.0
        feat["asymmetry_mean"] = 0.0
        return feat

    flipped_lr = cv2.flip(cropped, 1)
    xor_lr = cv2.bitwise_xor(cropped, flipped_lr)
    asym1 = np.count_nonzero(xor_lr) / lesion_area

    flipped_tb = cv2.flip(cropped, 0)
    xor_tb = cv2.bitwise_xor(cropped, flipped_tb)
    asym2 = np.count_nonzero(xor_tb) / lesion_area

    feat["asymmetry_axis1"] = round(asym1, 4)
    feat["asymmetry_axis2"] = round(asym2, 4)
    feat["asymmetry_mean"] = round((asym1 + asym2) / 2, 4)

    return feat


# ---------------------------------------------------------------------------
# B - Border
# ---------------------------------------------------------------------------
def compute_border_metrics(contour: np.ndarray) -> dict:
    """Helper to compute border irregularity given a specific contour."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return {"irregularity": 0.0, "compactness": 0.0}

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    pts = contour.reshape(-1, 2).astype(np.float64)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    irregularity = float(dists.std() / dists.mean()) if dists.mean() > 0 else 0.0
    
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    compactness = float(perim ** 2 / (4 * np.pi * area)) if area > 0 else 0.0
    
    return {"irregularity": irregularity, "compactness": compactness}


def compute_border(raw_contour: np.ndarray, light_contour: np.ndarray,
                   strong_contour: np.ndarray, mask: np.ndarray,
                   image_gray: np.ndarray) -> OrderedDict:
    """
    Multi-scale border irregularity analysis.
    """
    feat = OrderedDict()

    # Raw metrics (contains high frequency noise)
    raw_metrics = compute_border_metrics(raw_contour)
    
    # Light smoothing (removes pixel staircases)
    light_metrics = compute_border_metrics(light_contour)
    
    # Strong smoothing (base anatomical shape)
    strong_metrics = compute_border_metrics(strong_contour)
    
    # The actual reported border irregularity is derived from the LIGHT smoothing 
    # to preserve genuine structure but suppress artificial pixel-level staircases.
    feat["border_irregularity"] = round(light_metrics["irregularity"], 4)
    feat["compactness_index"] = round(light_metrics["compactness"], 4)
    
    # Noise ratio: How much irregularity is lost when applying strong smoothing?
    # If this is very high, the raw boundary was incredibly noisy.
    noise_ratio = raw_metrics["irregularity"] - strong_metrics["irregularity"]
    feat["segmentation_noise_factor"] = round(noise_ratio, 4)

    # Border sharpness: mean Sobel gradient along contour band
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask, k, iterations=2)
    eroded = cv2.erode(mask, k, iterations=2)
    border_band = cv2.subtract(dilated, eroded)

    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    border_pixels = grad_mag[border_band > 0]
    if len(border_pixels) > 0:
        feat["border_sharpness"] = round(float(border_pixels.mean()), 2)
    else:
        feat["border_sharpness"] = 0.0

    return feat


# ---------------------------------------------------------------------------
# C - Color
# ---------------------------------------------------------------------------
def compute_color(image_bgr: np.ndarray, mask: np.ndarray) -> OrderedDict:
    """
    Color features: k-means clustering, dominant colors, lesion vs skin.
    """
    feat = OrderedDict()

    lesion_px_bgr = image_bgr[mask > 0]
    if len(lesion_px_bgr) < 10:
        feat["n_colors"] = 0
        feat["dominant_colors"] = "N/A"
        feat["mean_rgb_lesion"] = "N/A"
        feat["mean_rgb_skin"] = "N/A"
        feat["mean_lab_lesion"] = "N/A"
        feat["std_rgb_lesion"] = "N/A"
        feat["std_lab_lesion"] = "N/A"
        return feat

    ml, sl = cv2.meanStdDev(image_bgr, mask=mask)
    feat["mean_rgb_lesion"] = f"({int(ml[2][0])}, {int(ml[1][0])}, {int(ml[0][0])})"
    feat["std_rgb_lesion"] = f"({int(sl[2][0])}, {int(sl[1][0])}, {int(sl[0][0])})"

    lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    ml_lab, sl_lab = cv2.meanStdDev(lab_img, mask=mask)
    feat["mean_lab_lesion"] = f"({int(ml_lab[0][0])}, {int(ml_lab[1][0])}, {int(ml_lab[2][0])})"
    feat["std_lab_lesion"] = f"({int(sl_lab[0][0])}, {int(sl_lab[1][0])}, {int(sl_lab[2][0])})"

    # Surrounding skin
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mask, k)
    surround = cv2.subtract(dilated, mask)
    if np.count_nonzero(surround) > 10:
        ms, _ = cv2.meanStdDev(image_bgr, mask=surround)
        feat["mean_rgb_skin"] = f"({int(ms[2][0])}, {int(ms[1][0])}, {int(ms[0][0])})"
    else:
        feat["mean_rgb_skin"] = "N/A"

    lesion_lab = lab_img[mask > 0].astype(np.float32)
    n_pixels = len(lesion_lab)
    k_clusters = min(6, n_pixels // 10)
    if k_clusters < 2:
        k_clusters = 2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        lesion_lab, k_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten(), minlength=k_clusters)
    total = counts.sum()
    significant = [(i, counts[i]) for i in range(k_clusters) if counts[i] / total > 0.05]

    feat["n_colors"] = len(significant)

    dominant = []
    for idx, count in sorted(significant, key=lambda x: -x[1]):
        pct = round(100 * count / total, 1)
        lab_color = centers[idx].reshape(1, 1, 3).astype(np.uint8)
        rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0, 0]
        dominant.append(
            f"({int(rgb_color[2])},{int(rgb_color[1])},{int(rgb_color[0])}) {pct}%"
        )
    feat["dominant_colors"] = "; ".join(dominant)

    return feat


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------
def extract_all_features(image_bgr: np.ndarray, contour: np.ndarray,
                         mask: np.ndarray, mm_per_px: float = None) -> OrderedDict:
    """
    Run the complete ABCDE feature extraction pipeline with advanced multi-scale smoothing.

    Parameters
    ----------
    image_bgr : np.ndarray  original BGR image
    contour   : np.ndarray  OpenCV contour (raw from segmentation)
    mask      : np.ndarray  uint8 binary mask (0/255)
    mm_per_px : float or None  scale factor

    Returns
    -------
    OrderedDict  all features grouped by category
    """
    result = OrderedDict()
    
    # 1. Generate smoothed contours to decouple true geometry from segmentation noise
    light_contour = smooth_contour(contour, num_points=300, smooth_factor=0.03)
    strong_contour = smooth_contour(contour, num_points=300, smooth_factor=0.10)

    # A — Geometry (calculated on lightly smoothed contour to avoid staircase artifact inflation)
    geom = compute_geometry(light_contour, mm_per_px)
    result.update(geom)

    # A — Asymmetry (calculated on mask directly)
    asym = compute_asymmetry(mask)
    result.update(asym)

    # B — Border (Multi-scale comparison)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bord = compute_border(contour, light_contour, strong_contour, mask, gray)
    result.update(bord)

    # C — Color
    col = compute_color(image_bgr, mask)
    result.update(col)

    # Inject smoothed contour into results for visualization
    result["debug_contours"] = {
        "raw": contour,
        "light_smooth": light_contour,
        "strong_smooth": strong_contour
    }

    return result
