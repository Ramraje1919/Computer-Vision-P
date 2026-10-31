
from pathlib import Path
import configparser

import cv2
import numpy as np
import pandas as pd
import motmetrics as mm

from ultralytics import YOLO
from tracker_baseline import IoUTracker


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

IOU_THRESHOLD = 0.20
MAX_MISSED = 15

MATCH_IOU_THRESHOLD = 0.50

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "baseline_evaluation_results.csv"


# =====================================================
# IOU FUNCTION
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
# LOAD SEQUENCE INFORMATION
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
# LOAD GROUND TRUTH
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

    gt = gt[
        (gt["class"] == 1) &
        (gt["confidence"] > 0)
    ].copy()

    return gt


# =====================================================
# MAIN EVALUATION
# =====================================================

def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_DIR}"
        )

    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {GT_PATH}"
        )

    info = load_sequence_info()

    print("\nSequence Information:")
    print(info)

    ground_truth = load_ground_truth()

    print(
        f"\nGround-truth annotations: {len(ground_truth)}"
    )

    print("\nLoading YOLO model...")

    model = YOLO(MODEL_NAME)

    # Baseline tracker
    tracker = IoUTracker(
        iou_threshold=IOU_THRESHOLD,
        max_missed=MAX_MISSED
    )

    accumulator = mm.MOTAccumulator(
        auto_id=True
    )

    total_frames = info["length"]

    print("\nStarting Baseline Evaluation...")

    for frame_number in range(1, total_frames + 1):

        image_path = (
            IMAGE_DIR /
            f"{frame_number:06d}{info['extension']}"
        )

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(
                f"Warning: Could not read frame {frame_number}"
            )
            continue

        # -------------------------------------------------
        # YOLO PERSON DETECTION
        # -------------------------------------------------

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

        # -------------------------------------------------
        # BASELINE TRACKER
        # -------------------------------------------------

        tracks = tracker.update(detections)

        # -------------------------------------------------
        # GROUND TRUTH FOR CURRENT FRAME
        # -------------------------------------------------

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

        # -------------------------------------------------
        # TRACK PREDICTIONS
        # -------------------------------------------------

        pred_ids = [
            track_id
            for track_id, box in tracks
        ]

        pred_boxes = []

        for track_id, box in tracks:

            x1, y1, x2, y2 = box

            pred_boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

        # -------------------------------------------------
        # IOU DISTANCE MATRIX
        # -------------------------------------------------

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

        accumulator.update(
            gt_ids,
            pred_ids,
            distances
        )

        if frame_number % 100 == 0:

            print(
                f"Evaluated {frame_number}/"
                f"{total_frames} frames"
            )

    # =====================================================
    # CALCULATE METRICS
    # =====================================================

    print("\nCalculating Baseline Metrics...")

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

    handler = mm.metrics.create()

    summary = handler.compute(
        accumulator,
        metrics=metrics,
        name="Baseline_IoU_Hungarian"
    )

    print("\n========== BASELINE RESULTS ==========")

    print(
        mm.io.render_summary(
            summary,
            formatters=handler.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
    )

    summary.to_csv(RESULT_PATH)

    print("\nBaseline Evaluation Completed!")

    print(
        f"Results saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
