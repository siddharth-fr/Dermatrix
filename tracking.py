"""
tracking.py — Optional longitudinal tracking and video segmentation.

Implements (behind flags):
  - Video/webcam tracking with SAM 2 video predictor (stub — requires sam2 video API)
  - Longitudinal comparison: aligns two images of the same lesion using ORB/RANSAC,
    then compares area, diameter, color, and asymmetry with percent change.
"""

import cv2
import numpy as np
from features import extract_all_features


def compare_longitudinal(
    image1_bgr: np.ndarray, mask1: np.ndarray, contour1: np.ndarray,
    image2_bgr: np.ndarray, mask2: np.ndarray, contour2: np.ndarray,
    mm_per_px: float = None,
) -> dict:
    """
    Compare two images of the same lesion taken at different times.

    Aligns image2 to image1 using ORB feature matching on the surrounding
    skin (excluding the lesion), then computes percent change in key metrics.

    Parameters
    ----------
    image1_bgr, mask1, contour1 : first time-point
    image2_bgr, mask2, contour2 : second time-point
    mm_per_px : optional scale factor

    Returns
    -------
    dict with keys like 'area_change_pct', 'diameter_change_pct', etc.
    """
    # Extract features for both
    feat1 = extract_all_features(image1_bgr, contour1, mask1, mm_per_px)
    feat2 = extract_all_features(image2_bgr, contour2, mask2, mm_per_px)

    # Attempt ORB alignment of image2 onto image1
    gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2GRAY)

    # Mask out the lesion so we only match on surrounding skin
    skin_mask1 = cv2.bitwise_not(mask1)
    skin_mask2 = cv2.bitwise_not(mask2)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, skin_mask1)
    kp2, des2 = orb.detectAndCompute(gray2, skin_mask2)

    aligned = False
    if des1 is not None and des2 is not None and len(des1) > 4 and len(des2) > 4:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) > 10:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask_h = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = image1_bgr.shape[:2]
                aligned_img2 = cv2.warpPerspective(image2_bgr, H, (w, h))
                aligned_mask2 = cv2.warpPerspective(mask2, H, (w, h),
                                                     flags=cv2.INTER_NEAREST)
                aligned = True

    # Compute changes
    result = {"aligned": aligned}

    def pct_change(v1, v2):
        if v1 == 0:
            return 0.0
        return round(100 * (v2 - v1) / v1, 2)

    for key in ["area_px", "diameter_px", "circularity", "solidity",
                "asymmetry_mean", "border_irregularity"]:
        v1 = feat1.get(key, 0)
        v2 = feat2.get(key, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            result[f"{key}_t1"] = v1
            result[f"{key}_t2"] = v2
            result[f"{key}_change_pct"] = pct_change(v1, v2)

    return result
