"""
Quantitative Evaluator for MOT17 Benchmark adhering to official MOTChallenge standards.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import motmetrics as mm

from src.evaluation.mot_format import load_mot_ground_truth
from src.tracking.association import box_ious


def evaluate_sequence(
    gt_file: Path,
    predictions_file: Path,
    seq_name: str = "MOT17-Seq",
    match_iou_thresh: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate tracker predictions against official MOT17 ground truth.
    Properly handles active target pedestrians vs ignore regions.
    """
    gt_df = load_mot_ground_truth(gt_file)

    # Active pedestrian targets
    active_gt = gt_df[(gt_df["class_id"] == 1) & (gt_df["conf"] > 0)].copy()

    # Ignore regions (static persons, distractors, persons on vehicles, reflections)
    ignore_gt = gt_df[(gt_df["class_id"] != 1) | (gt_df["conf"] == 0)].copy()

    # Load predictions
    pred_columns = ["frame", "id", "x", "y", "width", "height", "conf", "x3d", "y3d", "z3d"]
    if Path(predictions_file).exists() and Path(predictions_file).stat().st_size > 0:
        pred_df = pd.read_csv(predictions_file, header=None, names=pred_columns)
    else:
        pred_df = pd.DataFrame(columns=pred_columns)

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(list(set(active_gt["frame"].unique()) | set(pred_df["frame"].unique())))

    for frame_id in all_frames:
        frame_gt = active_gt[active_gt["frame"] == frame_id]
        frame_ignore = ignore_gt[ignore_gt["frame"] == frame_id]
        frame_pred = pred_df[pred_df["frame"] == frame_id]

        gt_ids = frame_gt["id"].astype(int).tolist()
        pred_ids = frame_pred["id"].astype(int).tolist()

        gt_boxes = frame_gt[["x", "y", "width", "height"]].to_numpy(dtype=float)
        # Convert xywh to xyxy for IoU computation
        gt_xyxy = np.column_stack([gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]]) if len(gt_boxes) > 0 else np.empty((0, 4))

        pred_boxes = frame_pred[["x", "y", "width", "height"]].to_numpy(dtype=float)
        pred_xyxy = np.column_stack([pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]]) if len(pred_boxes) > 0 else np.empty((0, 4))

        # Filter out predictions that overlap with ignore regions (e.g. static persons, distractor regions)
        valid_pred_mask = np.ones(len(pred_xyxy), dtype=bool)
        if len(frame_ignore) > 0 and len(pred_xyxy) > 0:
            ign_boxes = frame_ignore[["x", "y", "width", "height"]].to_numpy(dtype=float)
            ign_xyxy = np.column_stack([ign_boxes[:, 0], ign_boxes[:, 1], ign_boxes[:, 0] + ign_boxes[:, 2], ign_boxes[:, 1] + ign_boxes[:, 3]])
            ign_ious = box_ious(pred_xyxy, ign_xyxy)
            max_ign_iou = np.max(ign_ious, axis=1)
            valid_pred_mask = max_ign_iou < match_iou_thresh

        # Filtered predictions
        filtered_pred_ids = [pred_ids[i] for i in range(len(pred_ids)) if valid_pred_mask[i]]
        filtered_pred_xyxy = pred_xyxy[valid_pred_mask]

        if len(gt_xyxy) > 0 and len(filtered_pred_xyxy) > 0:
            ious = box_ious(gt_xyxy, filtered_pred_xyxy)
            distances = np.where(ious >= match_iou_thresh, 1.0 - ious, np.nan)
        else:
            distances = np.full((len(gt_ids), len(filtered_pred_ids)), np.nan, dtype=float)

        acc.update(gt_ids, filtered_pred_ids, distances)

    # Compute MOT metrics
    metrics = [
        "num_frames", "num_objects", "num_predictions", "num_matches",
        "num_misses", "num_false_positives", "num_switches", "num_fragmentations",
        "mostly_tracked", "mostly_lost", "partially_tracked",
        "mota", "motp", "idf1", "idp", "idr", "precision", "recall"
    ]

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=seq_name)

    row = summary.iloc[0].to_dict()
    # Format to clean serializable dictionary
    return {
        "sequence": seq_name,
        "mota": round(float(row.get("mota", 0.0)) * 100, 2),
        "idf1": round(float(row.get("idf1", 0.0)) * 100, 2),
        "idp": round(float(row.get("idp", 0.0)) * 100, 2),
        "idr": round(float(row.get("idr", 0.0)) * 100, 2),
        "idsw": int(row.get("num_switches", 0)),
        "frag": int(row.get("num_fragmentations", 0)),
        "fp": int(row.get("num_false_positives", 0)),
        "fn": int(row.get("num_misses", 0)),
        "precision": round(float(row.get("precision", 0.0)) * 100, 2),
        "recall": round(float(row.get("recall", 0.0)) * 100, 2),
        "mostly_tracked": int(row.get("mostly_tracked", 0)),
        "mostly_lost": int(row.get("mostly_lost", 0)),
        "num_matches": int(row.get("num_matches", 0)),
        "gt_objects": int(row.get("num_objects", 0)),
        "predictions": int(row.get("num_predictions", 0)),
    }

