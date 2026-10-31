"""
Postprocessing utilities: NMS, coordinate transformations, and boundary clipping.
"""

from typing import List, Tuple
import numpy as np


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h] (top-left width height)."""
    b = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    return np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=float)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    b = np.asarray(box, dtype=float)
    x, y, w, h = b[0], b[1], b[2], b[3]
    return np.array([x, y, x + w, y + h], dtype=float)


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to center coordinates [cx, cy, w, h]."""
    b = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=float)


def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    b = np.asarray(box, dtype=float)
    cx, cy, w, h = b[0], b[1], b[2], b[3]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip [x1, y1, x2, y2] boxes to image boundaries."""
    if len(boxes) == 0:
        return boxes
    b = np.asarray(boxes, dtype=float).copy()
    b[:, 0] = np.clip(b[:, 0], 0, width - 1)
    b[:, 1] = np.clip(b[:, 1], 0, height - 1)
    b[:, 2] = np.clip(b[:, 2], 0, width)
    b[:, 3] = np.clip(b[:, 3], 0, height)
    return b


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.65) -> List[int]:
    """
    Standard Non-Maximum Suppression (NMS) for [x1, y1, x2, y2] bounding boxes.
    Returns: List of integer indices to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

