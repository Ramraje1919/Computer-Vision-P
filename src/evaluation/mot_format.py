"""
MOTChallenge Data Format Serialization and Deserialization.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.detection.postprocessing import xyxy_to_xywh


def write_mot_results(results: List[Tuple[int, int, float, float, float, float, float]], output_path: Path):
    """
    Write tracking predictions in official MOTChallenge format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sorted(results, key=lambda x: (x[0], x[1])):
            frame, track_id, x, y, w, h, conf = item
            f.write(f"{int(frame)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")


def load_mot_ground_truth(gt_path: Path) -> pd.DataFrame:
    """
    Load official MOTChallenge gt.txt file.
    Filters: Pedestrian class (class 1) with active confidence (conf > 0).
    Non-pedestrian annotations (conf == 0 or class != 1) are flagged as ignore regions.
    """
    gt_path = Path(gt_path)
    columns = ["frame", "id", "x", "y", "width", "height", "conf", "class_id", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=columns)
    return df

