"""
MOT16-03.mp4 Full Video Tracking & Visualisation
=================================================
Reads the raw MP4 directly (no MOT17 sequence format).
Runs YOLO11n detection + ByteTrack (tuned params) + directional counter.
Saves annotated video and a tracking report to outputs/MOT16-03/.

NOTE: No quantitative metrics computed — no ground truth available for this file.
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.detector import Detection
from src.tracking.bytetrack import ByteTracker
from src.counting.line_counter import DirectionalLineCounter
from src.visualization.visualizer import MOTVisualizer
from src.utils.logger import get_logger

logger = get_logger("MOT16-03")

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_IN  = ROOT / "data" / "MOT16-03.mp4"
OUT_DIR   = ROOT / "outputs" / "MOT16-03"
VIDEO_OUT = OUT_DIR / "MOT16-03_tracked.mp4"
REPORT_MD = OUT_DIR / "tracking_report.md"
STATS_JSON= OUT_DIR / "tracking_stats.json"

# Tuned ByteTrack params (Stage H best config)
TRACKER_PARAMS = {
    "conf_high":           0.50,
    "conf_low":            0.10,
    "match_thresh_stage1": 0.80,
    "match_thresh_stage2": 0.45,
    "track_buffer":        90,
}

# YOLO detection config
YOLO_WEIGHTS  = str(ROOT / "yolo11n.pt")
YOLO_CONF     = 0.25          # lower threshold — YOLO filters internally
YOLO_CLASSES  = [0]           # class 0 = person
YOLO_IMGSZ    = 640


def load_yolo():
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    return model


def detect_frame(model, frame):
    """Run YOLO on a BGR frame, return list of [x1,y1,x2,y2,conf]."""
    results = model.predict(
        frame,
        conf=YOLO_CONF,
        classes=YOLO_CLASSES,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )
    dets = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            dets.append([x1, y1, x2, y2, conf])
    return dets


def det_to_track_input(dets):
    """Convert [x1,y1,x2,y2,conf] list to tracker Detection objects."""
    out = []
    for d in dets:
        x1, y1, x2, y2, conf = d
        out.append(Detection(box=[x1, y1, x2, y2], score=conf, class_id=0))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Open input video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        logger.error(f"Cannot open {VIDEO_IN}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Input : {VIDEO_IN}")
    logger.info(f"Size  : {width}x{height}  FPS={fps}  Frames={total}")
    logger.info(f"Output: {VIDEO_OUT}")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    # ── Tracker & counter ─────────────────────────────────────────────────────
    tracker   = ByteTracker(**TRACKER_PARAMS)
    y_line    = int(height * 0.55)
    counter   = DirectionalLineCounter(
        line_start=(60, y_line),
        line_end=(width - 60, y_line),
        cooldown_frames=30,
    )
    visualizer = MOTVisualizer()

    # ── YOLO model ────────────────────────────────────────────────────────────
    logger.info("Loading YOLO11n...")
    model = load_yolo()
    logger.info("YOLO loaded. Starting processing...")

    # ── Stats tracking ────────────────────────────────────────────────────────
    total_detections = 0
    total_tracks_seen = set()
    max_simultaneous  = 0
    frame_times       = []

    t_start = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.time()

        # Detect
        raw_dets = detect_frame(model, frame)
        total_detections += len(raw_dets)

        # Convert to tracker format
        dets = det_to_track_input(raw_dets)

        # Track
        tracks = tracker.update(dets)
        for t in tracks:
            total_tracks_seen.add(t.track_id)
        max_simultaneous = max(max_simultaneous, len(tracks))

        # Count
        counter.update(tracks, frame_id=frame_idx)

        # Visualise
        vis_frame = visualizer.draw_frame(
            frame, tracks,
            frame_idx=frame_idx,
            fps=fps,
            counter=counter,
        )
        writer.write(vis_frame)

        frame_times.append(time.time() - t0)

        if frame_idx % 100 == 0 or frame_idx == total:
            elapsed = time.time() - t_start
            avg_fps = frame_idx / elapsed
            eta     = (total - frame_idx) / avg_fps if avg_fps > 0 else 0
            counts  = counter.get_counts()
            logger.info(
                f"  Frame {frame_idx:4d}/{total} | "
                f"Tracks={len(tracks):2d} | "
                f"IN={counts['in']} OUT={counts['out']} | "
                f"Avg {avg_fps:.1f} fps | ETA {eta:.0f}s"
            )

    cap.release()
    writer.release()

    elapsed_total = time.time() - t_start
    counts = counter.get_counts()

    logger.info("=" * 60)
    logger.info(f"Done! Total time: {elapsed_total:.1f}s")
    logger.info(f"Unique track IDs   : {len(total_tracks_seen)}")
    logger.info(f"Max simultaneous   : {max_simultaneous}")
    logger.info(f"Total detections   : {total_detections}")
    logger.info(f"Counting — IN={counts['in']}  OUT={counts['out']}  TOTAL={counts['total']}")
    logger.info(f"Output video       : {VIDEO_OUT}")
    logger.info("=" * 60)

    # ── Save stats JSON ───────────────────────────────────────────────────────
    stats = {
        "video_in":          str(VIDEO_IN),
        "video_out":         str(VIDEO_OUT),
        "resolution":        f"{width}x{height}",
        "fps":               fps,
        "total_frames":      total,
        "duration_seconds":  round(total / fps, 1),
        "processing_time_s": round(elapsed_total, 1),
        "tracker_params":    TRACKER_PARAMS,
        "unique_track_ids":  len(total_tracks_seen),
        "max_simultaneous_tracks": max_simultaneous,
        "total_detections":  total_detections,
        "avg_dets_per_frame": round(total_detections / max(1, total), 2),
        "directional_counts": counts,
        "generated_at":      datetime.now().isoformat(),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved: {STATS_JSON}")

    # ── Save markdown report ──────────────────────────────────────────────────
    _write_report(stats)
    logger.info(f"Report saved: {REPORT_MD}")


def _write_report(s: dict):
    lines = [
        "# MOT16-03 — Full Video Tracking Report",
        "",
        f"> Generated: {s['generated_at'][:19].replace('T', ' ')}",
        f"> Tracker: ByteTrack (tuned params from Stage H)",
        f"> Detector: YOLO11n (conf ≥ 0.25, class=person)",
        "",
        "## Video Properties",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| File | `data/MOT16-03.mp4` |",
        f"| Resolution | {s['resolution']} |",
        f"| FPS | {s['fps']} |",
        f"| Total Frames | {s['total_frames']} |",
        f"| Duration | {s['duration_seconds']}s (~{s['duration_seconds']/60:.1f} min) |",
        f"| Processing Time | {s['processing_time_s']}s |",
        "",
        "## Tracker Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in s["tracker_params"].items():
        lines.append(f"| `{k}` | {v} |")

    c = s["directional_counts"]
    lines += [
        "",
        "## Tracking Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Unique track IDs assigned | {s['unique_track_ids']} |",
        f"| Max simultaneous tracks | {s['max_simultaneous_tracks']} |",
        f"| Total YOLO detections | {s['total_detections']} |",
        f"| Avg detections / frame | {s['avg_dets_per_frame']} |",
        "",
        "## Directional Counting",
        "",
        "A virtual counting line was placed at **55% frame height** "
        f"(y = {int(540 * 0.55)}px in a 960×540 frame).",
        "",
        "| Direction | Count |",
        "|-----------|-------|",
        f"| IN  (downward crossing) | **{c['in']}** |",
        f"| OUT (upward crossing)   | **{c['out']}** |",
        f"| Total crossings         | **{c['total']}** |",
        "",
        "## Output",
        "",
        f"Annotated video saved to: `outputs/MOT16-03/MOT16-03_tracked.mp4`",
        "",
        "> **Note**: No quantitative metrics (MOTA/IDF1) are computed for this video.",
        "> MOT16-03.mp4 is excluded from all benchmark evaluations by project protocol.",
        "",
        "---",
        "*Generated by `scripts/track_mot16_03.py`*",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
