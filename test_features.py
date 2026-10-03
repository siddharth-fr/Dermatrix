"""
test_features.py — Unit tests for features.py

Tests:
  - A perfect circle should have circularity ~1 and asymmetry ~0
  - An ellipse should have roundness < 1 and asymmetry > 0
  - A highly irregular shape should have high border irregularity
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import compute_geometry, compute_asymmetry, compute_border


def make_circle_mask(size=200, radius=60):
    """Create a perfect circle mask and contour."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours[0]


def make_ellipse_mask(size=200, axes=(80, 30)):
    """Create an ellipse mask and contour."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours[0]


def make_irregular_mask(size=200):
    """Create an irregular star-like shape."""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([
        [100, 20], [120, 70], [180, 70], [130, 100],
        [150, 160], [100, 120], [50, 160], [70, 100],
        [20, 70], [80, 70]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours[0]


def test_circle_circularity():
    mask, contour = make_circle_mask()
    geom = compute_geometry(contour, mask)
    circ = geom["circularity"]
    assert 0.85 < circ <= 1.0, f"Circle circularity should be ~1.0, got {circ}"
    print(f"  [PASS] Circle circularity = {circ:.4f}")


def test_circle_asymmetry():
    mask, _ = make_circle_mask()
    asym = compute_asymmetry(mask)
    mean_asym = asym["asymmetry_mean"]
    assert mean_asym < 0.1, f"Circle asymmetry should be ~0, got {mean_asym}"
    print(f"  [PASS] Circle asymmetry = {mean_asym:.4f}")


def test_ellipse_roundness():
    mask, contour = make_ellipse_mask()
    geom = compute_geometry(contour, mask)
    roundness = geom["roundness"]
    assert roundness < 0.6, f"Ellipse roundness should be < 0.6, got {roundness}"
    print(f"  [PASS] Ellipse roundness = {roundness:.4f}")


def test_ellipse_aspect_ratio():
    mask, contour = make_ellipse_mask(axes=(80, 30))
    geom = compute_geometry(contour, mask)
    ar = geom["aspect_ratio"]
    assert ar > 2.0, f"Ellipse aspect ratio should be > 2.0, got {ar}"
    print(f"  [PASS] Ellipse aspect ratio = {ar:.3f}")


def test_irregular_border():
    mask, contour = make_irregular_mask()
    gray = np.zeros((200, 200), dtype=np.uint8)
    gray[mask > 0] = 128
    bord = compute_border(contour, mask, gray)
    irreg = bord["border_irregularity"]
    assert irreg > 0.1, f"Irregular shape border irregularity should be > 0.1, got {irreg}"
    print(f"  [PASS] Irregular border irregularity = {irreg:.4f}")


def test_irregular_compactness():
    mask, contour = make_irregular_mask()
    gray = np.zeros((200, 200), dtype=np.uint8)
    bord = compute_border(contour, mask, gray)
    comp = bord["compactness_index"]
    assert comp > 1.2, f"Irregular compactness should be > 1.2, got {comp}"
    print(f"  [PASS] Irregular compactness = {comp:.4f}")


if __name__ == "__main__":
    tests = [
        test_circle_circularity,
        test_circle_asymmetry,
        test_ellipse_roundness,
        test_ellipse_aspect_ratio,
        test_irregular_border,
        test_irregular_compactness,
    ]

    print("Running feature tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: EXCEPTION - {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Passed: {passed}/{passed + failed}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
