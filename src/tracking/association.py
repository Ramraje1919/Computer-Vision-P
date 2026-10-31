"""
Association Metrics and Cost Matrices: IoU, Appearance Distance, and Spatial Gating.
"""

from typing import List, Optional
import numpy as np


def box_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of [x1, y1, x2, y2] boxes.
    Shape: (len(boxes_a), len(boxes_b))
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=float)
    boxes_b = np.asarray(boxes_b, dtype=float)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    )
    ih = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    )
    intersection = iw * ih
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU distance matrix: 1.0 - IoU."""
    return 1.0 - box_ious(boxes_a, boxes_b)


def fused_cost_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    appearance_distance: Optional[np.ndarray] = None,
    alpha: float = 0.40,
    spatial_gate_dist: float = 300.0,
    invalid_cost: float = 1e5
) -> np.ndarray:
    """
    Combine spatial IoU distance and appearance cosine distance with hard spatial gating.
    
    cost = alpha * C_iou + (1 - alpha) * C_app
    If center Euclidean distance > spatial_gate_dist, cost = invalid_cost.
    """
    c_iou = iou_distance(boxes_a, boxes_b)
    n_a, n_b = len(boxes_a), len(boxes_b)

    if n_a == 0 or n_b == 0:
        return c_iou

    if appearance_distance is not None and appearance_distance.shape == c_iou.shape:
        cost = alpha * c_iou + (1.0 - alpha) * appearance_distance
    else:
        cost = c_iou.copy()

    # Apply spatial gating
    centers_a = np.column_stack([(boxes_a[:, 0] + boxes_a[:, 2]) / 2.0, (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0])
    centers_b = np.column_stack([(boxes_b[:, 0] + boxes_b[:, 2]) / 2.0, (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0])

    diff = centers_a[:, None, :] - centers_b[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])

    # Where distance exceeds gate, set to invalid cost
    cost[dist > spatial_gate_dist] = invalid_cost

    return cost

