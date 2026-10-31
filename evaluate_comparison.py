
from pathlib import Path
import configparser

import cv2
import numpy as np
import pandas as pd
import motmetrics as mm

from ultralytics import YOLO

from tracker import IoUTracker as ImprovedTracker
from tracker_baseline import IoUTracker as BaselineTracker


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = (
    ROOT / "data" / "MOT17" / "train" / "MOT17-02-FRCNN"
)

IMAGE_DIR = SEQUENCE_DIR / "img1"
GT_PATH = SEQUENCE_DIR / "gt" / "gt.txt"
SEQINFO_PATH = SEQUENCE_DIR / "seqinfo.ini"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.20
IMAGE_SIZE = 960

MATCH_IOU_THRESHOLD = 0.50

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "tracker_comparison.csv"
RESULT_TXT = OUTPUT_DIR / "tracker_comparison.txt"


# =====================================================
# IOU
# =====================================================

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(
        box_a[0] + box_a[2],
        box_b[0] + box_b[2]
    )

    y2 = min(
        box_a[1] + box_a[3],
        box_b[1] + box_b[3]
    )

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# =====================================================
# SEQUENCE INFO
# =====================================================

def load_sequence_info():

    config = configparser.ConfigParser()
    config.read(SEQINFO_PATH)

    sequence = config["Sequence"]

    return {
        "fps": int(sequence["frameRate"]),
        "length": int(sequence["seqLength"]),
        "width": int(sequence["imWidth"]),
        "height": int(sequence["imHeight"]),
        "extension": sequence["imExt"]
    }


# =====================================================
# GROUND TRUTH
# =====================================================

def load_ground_truth():

    columns = [
        "frame",
        "id",
        "x",
        "y",
        "width",
        "height",
        "confidence",
        "class",
        "visibility"
    ]

    gt = pd.read_csv(
        GT_PATH,
        header=None,
        names=columns
    )

    # Keep pedestrian annotations only.
    gt = gt[
        (gt["class"] == 1) &
        (gt["confidence"] > 0)
    ].copy()

    return gt


# =====================================================
# BUILD DISTANCE MATRIX
# =====================================================

def build_distance_matrix(
    gt_boxes,
    pred_boxes
):

    distances = np.full(
        (len(gt_boxes), len(pred_boxes)),
        np.nan,
        dtype=float
    )

    for i, gt_box in enumerate(gt_boxes):

        for j, pred_box in enumerate(pred_boxes):

            overlap = calculate_iou(
                gt_box,
                pred_box
            )

            if overlap >= MATCH_IOU_THRESHOLD:
                distances[i, j] = 1.0 - overlap

    return distances


# =====================================================
# CONVERT TRACK BOXES
# =====================================================

def convert_tracks(tracks):

    track_ids = []
    boxes = []

    for track_id, box in tracks:

        x1, y1, x2, y2 = box

        track_ids.append(track_id)

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

    return track_ids, boxes


# =====================================================
# CALCULATE TRACKER METRICS
# =====================================================

def calculate_metrics(
    accumulator,
    tracker_name
):

    handler = mm.metrics.create()

    metrics = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_misses",
        "num_false_positives",
        "num_switches",
        "mota",
        "idf1",
        "idp",
        "idr"
    ]

    summary = handler.compute(
        accumulator,
        metrics=metrics,
        name=tracker_name
    )

    return summary


# =====================================================
# MAIN
# =====================================================

def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    if not GT_PATH.exists():
        raise FileNotFoundError(GT_PATH)

    info = load_sequence_info()

    ground_truth = load_ground_truth()

    print("\nSequence:", SEQUENCE_DIR.name)
    print("Frames:", info["length"])
    print("FPS:", info["fps"])
    print("Resolution:", info["width"], "x", info["height"])
    print("Ground truth pedestrian annotations:", len(ground_truth))

    print("\nLoading YOLO model...")

    model = YOLO(MODEL_NAME)

    # Independent tracker instances
    baseline_tracker = BaselineTracker(
        iou_threshold=0.20,
        max_missed=15
    )

    improved_tracker = ImprovedTracker(
        iou_threshold=0.10,
        max_missed=90,
        max_center_distance=150
    )

    baseline_acc = mm.MOTAccumulator(auto_id=True)
    improved_acc = mm.MOTAccumulator(auto_id=True)

    total_frames = info["length"]

    print("\nStarting comparison evaluation...")

    for frame_number in range(1, total_frames + 1):

        image_path = (
            IMAGE_DIR /
            f"{frame_number:06d}{info['extension']}"
        )

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Warning: Missing frame:", frame_number)
            continue

        # ---------------------------------------------
        # YOLO DETECTION (SHARED BY BOTH TRACKERS)
        # ---------------------------------------------

        results = model.predict(
            source=frame,
            classes=[0],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        detections = []

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            detections = (
                boxes.xyxy.cpu().numpy().tolist()
            )

        # ---------------------------------------------
        # RUN BOTH TRACKERS
        # ---------------------------------------------

        baseline_tracks = baseline_tracker.update(
            detections
        )

        improved_tracks = improved_tracker.update(
            detections
        )

        baseline_ids, baseline_boxes = convert_tracks(
            baseline_tracks
        )

        improved_ids, improved_boxes = convert_tracks(
            improved_tracks
        )

        # ---------------------------------------------
        # GROUND TRUTH FOR CURRENT FRAME
        # ---------------------------------------------

        gt_frame = ground_truth[
            ground_truth["frame"] == frame_number
        ]

        gt_ids = (
            gt_frame["id"]
            .astype(int)
            .tolist()
        )

        gt_boxes = gt_frame[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float)

        # ---------------------------------------------
        # DISTANCE MATRICES
        # ---------------------------------------------

        baseline_distances = build_distance_matrix(
            gt_boxes,
            baseline_boxes
        )

        improved_distances = build_distance_matrix(
            gt_boxes,
            improved_boxes
        )

        # ---------------------------------------------
        # UPDATE ACCUMULATORS
        # ---------------------------------------------

        baseline_acc.update(
            gt_ids,
            baseline_ids,
            baseline_distances
        )

        improved_acc.update(
            gt_ids,
            improved_ids,
            improved_distances
        )

        if frame_number % 100 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

    # =================================================
    # METRICS
    # =================================================

    print("\nCalculating final metrics...")

    baseline_summary = calculate_metrics(
        baseline_acc,
        "Baseline_IoU_Hungarian"
    )

    improved_summary = calculate_metrics(
        improved_acc,
        "Improved_Kalman_Hungarian"
    )

    comparison = pd.concat([
        baseline_summary,
        improved_summary
    ])

    print("\n========== FINAL COMPARISON ==========")

    handler = mm.metrics.create()

    print(
        mm.io.render_summary(
            comparison,
            formatters=handler.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
    )

    comparison.to_csv(RESULT_CSV)

    with open(
        RESULT_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MOT17 TRACKER COMPARISON\n")
        file.write("=" * 45 + "\n\n")

        file.write(
            comparison.to_string()
        )

        file.write("\n\nSequence: MOT17-02-FRCNN\n")
        file.write(f"Frames: {total_frames}\n")
        file.write(f"FPS: {info['fps']}\n")
        file.write(f"YOLO confidence: {CONFIDENCE}\n")
        file.write(f"YOLO image size: {IMAGE_SIZE}\n")

    print("\nComparison completed!")

    print("CSV:", RESULT_CSV)
    print("Report:", RESULT_TXT)


if __name__ == "__main__":
    main()
