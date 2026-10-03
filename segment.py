"""
segment.py — SAM 2 segmentation with intelligent mask selection and cleanup.

Loads a SAM 2 model once at import time and exposes:
  - estimate_box(image)   -> fallback bounding box when user doesn't draw one
  - segment(image, box)   -> validated, cleaned binary mask + quality metrics
"""

import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "mask_min_box_ratio": 0.02,   # Minimum area ratio of mask to bounding box
    "mask_max_box_ratio": 0.98,   # Maximum area ratio of mask to bounding box
    "max_border_touches": 2,      # Max number of image edges the mask can touch
    "min_solidity": 0.5,          # Minimum solidity (area / convex hull area)
    "min_area_px": 100,           # Minimum absolute mask area in pixels
}

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_predictor = None
_device = None
_model_loaded = False


def load_model():
    """
    Load SAM 2 model once.  Tries GPU first, falls back to CPU.
    Called at app startup.
    """
    global _predictor, _device, _model_loaded
    if _model_loaded:
        return

    import torch

    if torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"
    print(f"[segment] Using device: {_device}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        try:
            predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-large", device=_device
            )
            print("[segment] Loaded facebook/sam2.1-hiera-large")
        except Exception as e1:
            print(f"[segment] Large model failed ({e1}), trying base-plus...")
            try:
                predictor = SAM2ImagePredictor.from_pretrained(
                    "facebook/sam2.1-hiera-base-plus", device=_device
                )
                print("[segment] Loaded facebook/sam2.1-hiera-base-plus")
            except Exception as e2:
                print(f"[segment] base-plus also failed ({e2}), trying sam2 (v1)...")
                try:
                    predictor = SAM2ImagePredictor.from_pretrained(
                        "facebook/sam2-hiera-large", device=_device
                    )
                    print("[segment] Loaded facebook/sam2-hiera-large (v1)")
                except Exception as e3:
                    raise RuntimeError(
                        f"Could not load any SAM 2 model. "
                        f"Large: {e1}, Base-plus: {e2}, v1: {e3}"
                    ) from e3

        _predictor = predictor
        _model_loaded = True

    except ImportError as ie:
        raise RuntimeError("sam2 package not installed.") from ie


def get_device() -> str:
    return _device or "not loaded"


# ---------------------------------------------------------------------------
# Fallback box estimation
# ---------------------------------------------------------------------------
def estimate_box(image_rgb: np.ndarray) -> np.ndarray:
    """
    Automatically estimate a bounding box around the lesion.
    """
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Sample skin color from the center 50%
    center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
    center_x1, center_x2 = int(w * 0.25), int(w * 0.75)
    center_region = lab[center_y1:center_y2, center_x1:center_x2].reshape(-1, 3)
    
    valid_center = center_region[(center_region[:, 0] > 15) & (center_region[:, 0] < 240)]
    if len(valid_center) > 100:
        L_80 = np.percentile(valid_center[:, 0], 80)
        skin_pixels = valid_center[np.abs(valid_center[:, 0] - L_80) < 15]
        if len(skin_pixels) > 10:
            skin_median = np.median(skin_pixels, axis=0)
        else:
            skin_median = np.median(valid_center, axis=0)
    else:
        skin_median = np.median(center_region, axis=0)

    diff = np.linalg.norm(lab - skin_median, axis=2).astype(np.float32)
    diff_u8 = np.clip(diff * (255.0 / (diff.max() + 1e-5)), 0, 255).astype(np.uint8)

    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)])

    cx, cy = w / 2, h / 2
    best = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        mcx = M["m10"] / M["m00"]
        mcy = M["m01"] / M["m00"]
        dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        centrality = 1.0 - dist / max_dist
        score = area * centrality
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return np.array([int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)])

    x, y, bw, bh = cv2.boundingRect(best)
    mx = int(bw * 0.10)
    my = int(bh * 0.10)

    if bw > w * 0.9 and bh > h * 0.9:
        return np.array([int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)])

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)

    return np.array([x1, y1, x2, y2])


# ---------------------------------------------------------------------------
# Mask validation and Selection
# ---------------------------------------------------------------------------
def _evaluate_mask_quality(mask: np.ndarray, box: np.ndarray) -> dict:
    """
    Evaluate a mask's anatomical plausibility and structural validity.
    Returns a dictionary of metrics and a final quality status string.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    box_area = bw * bh
    
    h, w = mask.shape
    bx1 = max(0, int(x1))
    by1 = max(0, int(y1))
    bx2 = min(w, int(x2))
    by2 = min(h, int(y2))
    
    mask_area = np.count_nonzero(mask)
    if mask_area < CONFIG["min_area_px"]:
        return {"status": "FAILED", "reason": "Too small", "box_ratio": 0, "solidity": 0, "components": 0}

    cropped = mask[by1:by2, bx1:bx2]
    area_in_box = np.count_nonzero(cropped)
    box_ratio = area_in_box / (box_area + 1e-5)
    
    # Border touches
    sides_touching = 0
    if np.any(mask[0, :]): sides_touching += 1
    if np.any(mask[-1, :]): sides_touching += 1
    if np.any(mask[:, 0]): sides_touching += 1
    if np.any(mask[:, -1]): sides_touching += 1

    # Solidity and Components
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "FAILED", "reason": "No contours", "box_ratio": box_ratio, "solidity": 0, "components": 0}
        
    largest_cnt = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_cnt)
    hull = cv2.convexHull(largest_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = largest_area / (hull_area + 1e-5)
    
    status = "GOOD"
    reasons = []
    
    if box_ratio < CONFIG["mask_min_box_ratio"]:
        status = "FAILED"
        reasons.append("Fills <2% of box")
    elif box_ratio > CONFIG["mask_max_box_ratio"]:
        status = "QUESTIONABLE"
        reasons.append("Fills almost entire box")
        
    if sides_touching > CONFIG["max_border_touches"]:
        status = "FAILED"
        reasons.append(f"Touches {sides_touching} borders")
        
    if solidity < CONFIG["min_solidity"]:
        status = "QUESTIONABLE"
        reasons.append(f"Low solidity ({solidity:.2f})")
        
    if len(contours) > 3:
        status = "QUESTIONABLE"
        reasons.append(f"Highly fragmented ({len(contours)} pieces)")
        
    return {
        "status": status,
        "reason": ", ".join(reasons) if reasons else "Pass",
        "box_ratio": box_ratio,
        "solidity": solidity,
        "components": len(contours),
        "mask_area": mask_area,
        "largest_area": largest_area
    }


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean the mask by removing tiny isolated components and filling holes.
    AVOIDS harsh morphological opening/closing which destroys genuine edge irregularity.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    
    # 1. Keep only the largest connected component (and any significantly large secondary ones)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels > 1:
        # label 0 is background
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = np.max(areas)
        clean = np.zeros_like(mask_u8)
        # Keep components that are at least 15% of the max component size
        for i, area in enumerate(areas):
            if area > max_area * 0.15:
                clean[labels == i + 1] = 255
    else:
        clean = mask_u8.copy()

    # 2. Fill holes inside the components
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(clean, contours, -1, 255, -1)

    return clean


# ---------------------------------------------------------------------------
# Main segmentation entry point
# ---------------------------------------------------------------------------
def segment(image_rgb: np.ndarray, box: np.ndarray = None) -> dict:
    """
    Segment a lesion using SAM 2.
    """
    import torch

    if not _model_loaded or _predictor is None:
        raise RuntimeError("SAM 2 model not loaded. Call segment.load_model() first.")

    if box is None:
        box = estimate_box(image_rgb)

    box_np = np.array(box, dtype=np.float32)
    cx = (box_np[0] + box_np[2]) / 2.0
    cy = (box_np[1] + box_np[3]) / 2.0
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    with torch.inference_mode():
        _predictor.set_image(image_rgb)
        raw_masks, scores, _ = _predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            multimask_output=True,
        )

    # Intelligent Mask Selection
    # Score masks based on SAM confidence + Plausibility
    best_candidate = None
    best_plausibility = -9999.0
    
    candidates = []
    
    for idx, (m, score) in enumerate(zip(raw_masks, scores)):
        quality = _evaluate_mask_quality(m, box_np)
        
        # Calculate Plausibility Score
        # Start with SAM's score (typically 0.7 to 1.0)
        p_score = float(score)
        
        if quality["status"] == "FAILED":
            p_score -= 10.0  # Heavily penalize failed masks
        elif quality["status"] == "QUESTIONABLE":
            p_score -= 0.5   # Moderately penalize questionable masks
            
        # Reward solid, cohesive masks
        p_score += (quality.get("solidity", 0) * 0.2)
        
        candidates.append({
            "mask": m,
            "sam_score": float(score),
            "quality": quality,
            "p_score": p_score,
            "is_raw": True
        })
        
        if p_score > best_plausibility:
            best_plausibility = p_score
            best_candidate = candidates[-1]

    # If all failed, retry with enlarged box
    if best_candidate["quality"]["status"] == "FAILED":
        h, w = image_rgb.shape[:2]
        bw = box_np[2] - box_np[0]
        bh = box_np[3] - box_np[1]
        enlarged = np.array([
            max(0, box_np[0] - bw * 0.15),
            max(0, box_np[1] - bh * 0.15),
            min(w, box_np[2] + bw * 0.15),
            min(h, box_np[3] + bh * 0.15),
        ], dtype=np.float32)

        with torch.inference_mode():
            masks2, scores2, _ = _predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=enlarged,
                multimask_output=True,
            )

        for m, score in zip(masks2, scores2):
            quality = _evaluate_mask_quality(m, enlarged)
            p_score = float(score)
            if quality["status"] == "FAILED": p_score -= 10.0
            elif quality["status"] == "QUESTIONABLE": p_score -= 0.5
            p_score += (quality.get("solidity", 0) * 0.2)
            
            if p_score > best_plausibility:
                best_plausibility = p_score
                best_candidate = {
                    "mask": m,
                    "sam_score": float(score),
                    "quality": quality,
                    "p_score": p_score,
                    "is_raw": True
                }
                box_np = enlarged

    # Cleanup the best mask
    raw_mask = best_candidate["mask"]
    clean_mask = _clean_mask(raw_mask)
    
    # Re-evaluate final cleaned mask
    final_quality = _evaluate_mask_quality(clean_mask, box_np)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to raw if cleanup destroyed it completely
        clean_mask = (raw_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Segmentation failed to produce a valid contour.")

    largest_contour = max(contours, key=cv2.contourArea)

    return {
        "mask": clean_mask,           # Final cleaned binary mask (uint8)
        "raw_mask": (raw_mask > 0).astype(np.uint8) * 255, # Raw SAM mask for debugging
        "contour": largest_contour,   # OpenCV contour of cleaned mask
        "box": box_np.astype(int),
        "score": best_candidate["sam_score"],
        "segmentation_quality": final_quality["status"],
        "quality_metrics": final_quality
    }
