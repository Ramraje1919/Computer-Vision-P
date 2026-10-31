"""
Preprocessing routines for object detection.
"""

from typing import Tuple
import cv2
import numpy as np


def resize_letterbox(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (960, 960),
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterbox padding to preserve aspect ratio.
    Returns: (padded_image, scale_ratio, (pad_left, pad_top))
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, (pad_left, pad_top)

