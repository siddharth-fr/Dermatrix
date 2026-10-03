"""
evaluate.py — Evaluation script for lesion segmentation accuracy.

Takes a folder of images + ground-truth masks (ISIC 2018 / PH2 format),
auto-generates a box from each ground-truth mask (with random 5-15% jitter),
runs the pipeline, and reports mean Dice, IoU, and saves the 10 worst cases.

Usage:
    python evaluate.py --images ./data/images --masks ./data/masks --out ./eval_results
"""

import os
import argparse
import random
import json
import cv2
import numpy as np

from preprocess import full_preprocess
from segment import load_model, segment


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    intersection = np.count_nonzero(pred_b & gt_b)
    total = np.count_nonzero(pred_b) + np.count_nonzero(gt_b)
    if total == 0:
        return 1.0
    return 2 * intersection / total


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    intersection = np.count_nonzero(pred_b & gt_b)
    union = np.count_nonzero(pred_b | gt_b)
    if union == 0:
        return 1.0
    return intersection / union


def box_from_gt_mask(mask: np.ndarray, jitter_pct: float = None) -> np.ndarray:
    """
    Derive a bounding box from a ground-truth mask, with optional random jitter.

    Parameters
    ----------
    mask : np.ndarray  binary (0/255)
    jitter_pct : float or None  random jitter between 5-15% if None

    Returns
    -------
    np.ndarray  [x1, y1, x2, y2]
    """
    if jitter_pct is None:
        jitter_pct = random.uniform(0.05, 0.15)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return np.array([0, 0, w, h])

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add jitter
    jx = int(w * jitter_pct)
    jy = int(h * jitter_pct)
    ih, iw = mask.shape

    x1 = max(0, x - jx + random.randint(-jx // 2, jx // 2))
    y1 = max(0, y - jy + random.randint(-jy // 2, jy // 2))
    x2 = min(iw, x + w + jx + random.randint(-jx // 2, jx // 2))
    y2 = min(ih, y + h + jy + random.randint(-jy // 2, jy // 2))

    return np.array([x1, y1, x2, y2])


def main():
    parser = argparse.ArgumentParser(description="Evaluate lesion segmentation")
    parser.add_argument("--images", required=True, help="Folder of input images")
    parser.add_argument("--masks", required=True, help="Folder of ground-truth masks")
    parser.add_argument("--out", default="./eval_results", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model
    print("Loading SAM 2 model...")
    load_model()

    # Collect image-mask pairs
    image_files = sorted([
        f for f in os.listdir(args.images)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    results = []
    for fname in image_files:
        img_path = os.path.join(args.images, fname)
        # Try common mask naming patterns
        base = os.path.splitext(fname)[0]
        mask_path = None
        for suffix in ['_segmentation.png', '_mask.png', '.png', '_segmentation.bmp']:
            candidate = os.path.join(args.masks, base + suffix)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"  Skipping {fname}: no matching mask found")
            continue

        img = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask is None:
            print(f"  Skipping {fname}: could not read")
            continue

        # Resize mask if needed
        if gt_mask.shape[:2] != img.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Binarize GT mask
        _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

        # Generate box from GT
        box = box_from_gt_mask(gt_mask)

        # Preprocess
        preprocessed = full_preprocess(img)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        try:
            result = segment(rgb, box=box)
            pred_mask = result["mask"]

            d = dice_score(pred_mask, gt_mask)
            i = iou_score(pred_mask, gt_mask)
            results.append({"file": fname, "dice": d, "iou": i})
            print(f"  {fname}: Dice={d:.4f}  IoU={i:.4f}")

        except Exception as e:
            results.append({"file": fname, "dice": 0.0, "iou": 0.0, "error": str(e)})
            print(f"  {fname}: ERROR — {e}")

    if not results:
        print("No valid image-mask pairs found.")
        return

    # Summary
    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    print(f"\n{'='*50}")
    print(f"Results: {len(results)} images")
    print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Mean IoU:  {np.mean(ious):.4f} ± {np.std(ious):.4f}")

    # Save worst 10
    sorted_results = sorted(results, key=lambda r: r["dice"])
    worst_10 = sorted_results[:10]

    print(f"\nWorst 10 cases:")
    for r in worst_10:
        print(f"  {r['file']}: Dice={r['dice']:.4f}")
        # Save overlay
        img = cv2.imread(os.path.join(args.images, r["file"]))
        gt_mask = None
        base = os.path.splitext(r["file"])[0]
        for suffix in ['_segmentation.png', '_mask.png', '.png']:
            candidate = os.path.join(args.masks, base + suffix)
            if os.path.exists(candidate):
                gt_mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                break

        if img is not None and gt_mask is not None:
            if gt_mask.shape[:2] != img.shape[:2]:
                gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            overlay = img.copy()
            overlay[gt_mask > 0] = (overlay[gt_mask > 0] * 0.5 +
                                     np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            out_path = os.path.join(args.out, f"worst_{r['file']}")
            cv2.imwrite(out_path, overlay)

    # Save JSON report
    report = {
        "n_images": len(results),
        "mean_dice": round(float(np.mean(dices)), 4),
        "std_dice": round(float(np.std(dices)), 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        "std_iou": round(float(np.std(ious)), 4),
        "per_image": results,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {os.path.join(args.out, 'report.json')}")


if __name__ == "__main__":
    main()
