import cv2
import numpy as np
import torch
from PIL import Image
import traceback

_depth_estimator = None

def get_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[depth] Loading Intel/dpt-large on {device}...")
            _depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)
            print("[depth] Model loaded successfully.")
        except Exception as e:
            print(f"[depth] Error loading depth model: {e}")
            raise e
    return _depth_estimator

def estimate_depth(bgr_image, mask=None):
    """
    Given a BGR image crop, estimate relative depth.
    Returns a normalized grayscale depth map (uint8, 0-255).
    """
    try:
        estimator = get_depth_estimator()
        
        # Convert to RGB PIL Image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        
        # Run depth estimation
        result = estimator(pil_img)
        depth_pil = result["depth"]
        
        depth_map = np.array(depth_pil)
        
        # Resize depth map back to original crop size
        h, w = bgr_image.shape[:2]
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-255 uint8
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            depth_norm = (depth_norm * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
            
        # Optional: apply mask so background is flat
        if mask is not None:
            if mask.shape != depth_norm.shape:
                mask = cv2.resize(mask, (depth_norm.shape[1], depth_norm.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Find the mean depth of the edge of the mask to use as the background level
            # For simplicity, we just set background to 0 (lowest depth)
            depth_norm[mask == 0] = 0
            
        return depth_norm
    except Exception as e:
        traceback.print_exc()
        return None
