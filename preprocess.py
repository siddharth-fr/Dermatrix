"""
preprocess.py — Image preprocessing for lesion analysis.

Provides:
  - Shades-of-Gray color constancy (Minkowski p=6)
  - DullRazor hair removal (blackhat morphology + inpaint)
  - Glare inpainting (near-white saturated pixel removal)
"""

import cv2
import numpy as np


def shades_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Shades-of-Gray color constancy (Minkowski norm).

    Estimates the scene illuminant using the p-norm of each color channel,
    then normalises so the image looks as if taken under a white light source.

    Parameters
    ----------
    image : np.ndarray  BGR uint8
    p : int             Minkowski exponent (6 is a good default)

    Returns
    -------
    np.ndarray  BGR uint8, colour-corrected
    """
    img = image.astype(np.float64)
    # Per-channel p-norm
    norms = np.zeros(3)
    for c in range(3):
        ch = img[:, :, c]
        norms[c] = np.power(np.mean(np.power(ch, p)), 1.0 / p) + 1e-10
    # Scale so the estimated illuminant becomes white
    max_norm = norms.max()
    gains = max_norm / norms
    out = np.clip(img * gains[np.newaxis, np.newaxis, :], 0, 255)
    return out.astype(np.uint8)


def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    DullRazor-style hair removal.

    Uses morphological blackhat to detect dark thin structures (hairs),
    thresholds them, and inpaints the result.

    Parameters
    ----------
    image : np.ndarray  BGR uint8

    Returns
    -------
    np.ndarray  BGR uint8, hairs removed
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def remove_glare(image: np.ndarray) -> np.ndarray:
    """
    Detect near-white, high-saturation pixels (specular glare) and inpaint.

    Parameters
    ----------
    image : np.ndarray  BGR uint8

    Returns
    -------
    np.ndarray  BGR uint8, glare removed
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Near-white: high value, low saturation
    glare_mask = cv2.inRange(hsv, (0, 0, 240), (180, 30, 255))
    # Also catch blown-out pixels in grayscale
    _, bright = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    glare_mask = cv2.bitwise_or(glare_mask, bright)
    # Dilate slightly to cover edges of glare spots
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.dilate(glare_mask, k, iterations=1)
    if np.count_nonzero(glare_mask) == 0:
        return image
    return cv2.inpaint(image, glare_mask, 5, cv2.INPAINT_TELEA)


def full_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Run the complete preprocessing pipeline:
      1. Color constancy
      2. Hair removal
      3. Glare removal

    Parameters
    ----------
    image : np.ndarray  BGR uint8

    Returns
    -------
    np.ndarray  BGR uint8, preprocessed
    """
    img = shades_of_gray(image)
    img = remove_hair(img)
    img = remove_glare(img)
    return img
