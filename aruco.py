"""
aruco.py — ArUco marker detection and geometric calibration.

Detects ArUco markers in the image to establish:
  - mm_per_px: real-world scale from known marker physical size
  - Perspective correction: dewarp the image to remove tilt/angle distortion
  - Tilt / viewing angle estimates

The markers have a known physical size of 13.5mm (1.35cm) per side.
"""

import cv2
import numpy as np
from collections import OrderedDict

# Physical side length of the ArUco marker in millimeters
MARKER_PHYSICAL_SIZE_MM = 13.5


def detect_markers(image_bgr: np.ndarray, dictionary_name: str = "DICT_4X4_250"):
    """
    Detect ArUco markers in a BGR image.

    Parameters
    ----------
    image_bgr : np.ndarray  BGR uint8
    dictionary_name : str    OpenCV ArUco dictionary name

    Returns
    -------
    list of dict, each with:
        'id'      : int          marker ID
        'corners' : np.ndarray   (4, 2) float32 — four corner points in image pixels
                                 Order: top-left, top-right, bottom-right, bottom-left
    """
    # Get the ArUco dictionary
    dict_id = getattr(cv2.aruco, dictionary_name, None)
    if dict_id is None:
        # Fallback: try common dictionaries
        for fallback in ["DICT_4X4_250", "DICT_4X4_100", "DICT_4X4_50",
                         "DICT_5X5_250", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL"]:
            dict_id = getattr(cv2.aruco, fallback, None)
            if dict_id is not None:
                break
        if dict_id is None:
            return []

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()

    # Tune detection for various conditions
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.01   # Allow small markers in big images
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image_bgr)

    results = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]  # shape (4, 2)
            results.append({
                "id": int(marker_id),
                "corners": pts.astype(np.float32),
            })

    return results


def compute_marker_geometry(corners: np.ndarray) -> dict:
    """
    From 4 corner points of a detected marker, compute geometric properties.

    Parameters
    ----------
    corners : np.ndarray  (4, 2) — the four corners

    Returns
    -------
    dict with:
        'side_lengths_px'  : list of 4 float  — pixel length of each side
        'mean_side_px'     : float             — average side length in pixels
        'mm_per_px'        : float             — scale factor
        'px_per_mm'        : float             — inverse scale factor
        'tilt_angle_deg'   : float             — estimated tilt from camera axis
        'rotation_deg'     : float             — in-plane rotation of the marker
        'area_px'          : float             — area of the quadrilateral
        'aspect_ratio'     : float             — ratio of opposing side means
    """
    pts = corners.astype(np.float64)

    # Side lengths
    sides = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        sides.append(np.linalg.norm(p2 - p1))

    mean_side = np.mean(sides)
    mm_per_px = MARKER_PHYSICAL_SIZE_MM / mean_side if mean_side > 0 else 0

    # Perspective tilt estimation
    # A perfect fronto-parallel square has all sides equal.
    # The ratio of shortest to longest side indicates tilt.
    min_side = min(sides)
    max_side = max(sides)
    foreshortening = min_side / (max_side + 1e-8)
    # Approximate tilt angle: cos(theta) ≈ foreshortening
    tilt_angle_rad = np.arccos(np.clip(foreshortening, 0, 1))
    tilt_angle_deg = np.degrees(tilt_angle_rad)

    # Also estimate tilt from the ratio of opposing side averages
    side_01 = (sides[0] + sides[2]) / 2  # top+bottom
    side_12 = (sides[1] + sides[3]) / 2  # left+right
    aspect_ratio = max(side_01, side_12) / (min(side_01, side_12) + 1e-8)

    # In-plane rotation: angle of the top edge relative to horizontal
    top_edge = pts[1] - pts[0]
    rotation_deg = np.degrees(np.arctan2(top_edge[1], top_edge[0]))

    # Quadrilateral area (Shoelace formula)
    n = 4
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    area = abs(area) / 2

    return {
        "side_lengths_px": [round(s, 1) for s in sides],
        "mean_side_px": round(mean_side, 2),
        "mm_per_px": round(mm_per_px, 5),
        "px_per_mm": round(1.0 / mm_per_px, 2) if mm_per_px > 0 else 0,
        "tilt_angle_deg": round(tilt_angle_deg, 1),
        "rotation_deg": round(rotation_deg, 1),
        "area_px": round(area, 1),
        "aspect_ratio": round(aspect_ratio, 3),
    }


def compute_perspective_correction(
    image_bgr: np.ndarray, corners: np.ndarray
) -> tuple:
    """
    Compute the homography that dewarps the image so the marker becomes
    a perfect square, then apply it to the full image.

    Parameters
    ----------
    image_bgr : np.ndarray  original BGR image
    corners   : np.ndarray  (4, 2) detected marker corners

    Returns
    -------
    (corrected_image, H, mm_per_px_corrected)
        corrected_image : np.ndarray  perspective-corrected BGR image
        H               : np.ndarray  3x3 homography matrix
        mm_per_px_corr  : float       scale in the corrected image
    """
    h, w = image_bgr.shape[:2]
    src_pts = corners.astype(np.float32)

    # Compute where the marker SHOULD be if viewed fronto-parallel.
    # We place it at the same centroid but with a perfect square of the
    # average side length.
    mean_side = np.mean([np.linalg.norm(src_pts[(i+1)%4] - src_pts[i])
                         for i in range(4)])
    cx = np.mean(src_pts[:, 0])
    cy = np.mean(src_pts[:, 1])
    half = mean_side / 2

    dst_pts = np.array([
        [cx - half, cy - half],  # top-left
        [cx + half, cy - half],  # top-right
        [cx + half, cy + half],  # bottom-right
        [cx - half, cy + half],  # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply to full image
    corrected = cv2.warpPerspective(image_bgr, H, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    # In the corrected image, the marker has mean_side px per 13.5mm
    mm_per_px_corrected = MARKER_PHYSICAL_SIZE_MM / mean_side if mean_side > 0 else 0

    return corrected, H, mm_per_px_corrected


def transform_contour(contour: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a homography to an OpenCV contour.

    Parameters
    ----------
    contour : np.ndarray  shape (N, 1, 2) — standard OpenCV contour
    H       : np.ndarray  3x3 homography (forward: original → corrected)

    Returns
    -------
    np.ndarray  transformed contour, same shape
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    ones = np.ones((len(pts), 1))
    homog = np.hstack([pts, ones])  # (N, 3)
    transformed = (H @ homog.T).T  # (N, 3)
    w = transformed[:, 2:3]
    w[w == 0] = 1e-8
    xy = transformed[:, :2] / w
    return xy.astype(np.int32).reshape(-1, 1, 2)


def transform_mask(mask: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a homography to a binary mask.
    """
    h, w = mask.shape[:2]
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)


def calibrate(image_bgr: np.ndarray) -> dict:
    """
    Full ArUco calibration pipeline.

    Detects markers, picks the best one, computes geometry and
    perspective correction.

    Parameters
    ----------
    image_bgr : np.ndarray  BGR image

    Returns
    -------
    dict with:
        'found'             : bool
        'markers'           : list of detected markers
        'best_marker'       : dict with id + corners (or None)
        'geometry'          : dict from compute_marker_geometry (or None)
        'mm_per_px'         : float (or None)
        'corrected_image'   : np.ndarray (or None — original image if no markers)
        'corrected_mm_per_px': float (scale in corrected image, or None)
        'homography'        : np.ndarray 3x3 (or None)
    """
    # Try multiple dictionaries
    markers = []
    for dict_name in ["DICT_4X4_250", "DICT_4X4_100", "DICT_4X4_50",
                      "DICT_5X5_250", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL"]:
        markers = detect_markers(image_bgr, dict_name)
        if markers:
            break

    if not markers:
        return {
            "found": False,
            "markers": [],
            "best_marker": None,
            "geometry": None,
            "mm_per_px": None,
            "corrected_image": None,
            "corrected_mm_per_px": None,
            "homography": None,
        }

    # Pick the best marker: largest area = most reliable
    best = max(markers, key=lambda m: cv2.contourArea(m["corners"].reshape(-1, 1, 2)))

    geom = compute_marker_geometry(best["corners"])

    # Compute perspective correction
    corrected_img, H, corrected_mm_per_px = compute_perspective_correction(
        image_bgr, best["corners"]
    )

    return {
        "found": True,
        "markers": markers,
        "best_marker": best,
        "geometry": geom,
        "mm_per_px": geom["mm_per_px"],
        "corrected_image": corrected_img,
        "corrected_mm_per_px": corrected_mm_per_px,
        "homography": H,
    }


def get_calibration_summary(cal: dict) -> OrderedDict:
    """
    Return a human-readable summary of calibration results for the UI.
    """
    summary = OrderedDict()

    if not cal["found"]:
        summary["aruco_status"] = "No marker detected"
        summary["scale"] = "N/A (manual entry or unknown)"
        return summary

    g = cal["geometry"]
    summary["aruco_status"] = f"Marker ID {cal['best_marker']['id']} detected"
    summary["markers_found"] = len(cal["markers"])
    summary["marker_side_px"] = f"{g['mean_side_px']} px"
    summary["scale_mm_per_px"] = f"{g['mm_per_px']:.4f}"
    summary["scale_px_per_mm"] = f"{g['px_per_mm']:.1f}"
    summary["tilt_angle"] = f"{g['tilt_angle_deg']}deg"
    summary["rotation"] = f"{g['rotation_deg']}deg"
    summary["perspective"] = "Corrected" if g["tilt_angle_deg"] > 3.0 else "Near fronto-parallel"

    return summary
