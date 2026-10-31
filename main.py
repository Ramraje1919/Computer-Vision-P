
from pathlib import Path
import cv2
import csv
from ultralytics import YOLO

# Support both `python -m src.main` and `python src/main.py`.
if __package__:
    from .tracker_reid import IoUTracker
else:
    from tracker_reid import IoUTracker


# =====================================================
# CONFIGURATION
# =====================================================

VIDEO_PATH = Path("data/MOT16-03.mp4")

OUTPUT_PATH = Path("outputs/annotated_reid_output.mp4")

EVENT_LOG_PATH = Path("reports/reid_tracking_events.csv")

# Per-frame visible track coordinates for identity-switch diagnosis.
TRAJECTORY_LOG_PATH = Path("reports/reid_tracking_trajectories.csv")

SUMMARY_PATH = Path("reports/reid_tracking_summary.txt")

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.20
IMAGE_SIZE = 960

IOU_THRESHOLD = 0.10
MAX_MISSED = 150
MAX_CENTER_DISTANCE = 350
APPEARANCE_THRESHOLD = 0.25

# Set to 300 for testing.
# Set to None to process the complete video.
MAX_FRAMES = 600


# =====================================================
# MAIN
# =====================================================

def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    EVENT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAJECTORY_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading YOLO model...")

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 15.0

    frames_to_process = (
        min(total_frames, MAX_FRAMES)
        if MAX_FRAMES is not None
        else total_frames
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # =================================================
    # INITIALIZE RE-ID TRACKER
    # =================================================

    tracker = IoUTracker(
        iou_threshold=IOU_THRESHOLD,
        max_missed=MAX_MISSED,
        max_center_distance=MAX_CENTER_DISTANCE,
        appearance_threshold=APPEARANCE_THRESHOLD
    )

    frame_number = 0

    total_detections = 0
    total_track_observations = 0

    event_counts = {
        "NEW_TRACK": 0,
        "TRACK_MISSING": 0,
        "TRACK_RECOVERED": 0,
        "TRACK_EXPIRED": 0
    }

    print("\nStarting Appearance-based Re-ID tracking...")
    print(f"Frames to process: {frames_to_process}")

    trajectory_file = open(
        TRAJECTORY_LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    )
    trajectory_writer = csv.writer(trajectory_file)
    trajectory_writer.writerow([
        "frame", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2"
    ])

    try:

        with open(
            EVENT_LOG_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame",
                "timestamp_sec",
                "event",
                "track_id",
                "details"
            ])

            while True:

                # Stop after configured frame limit.
                if (
                    MAX_FRAMES is not None
                    and frame_number >= MAX_FRAMES
                ):
                    break

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                timestamp = frame_number / fps

                # -------------------------------------
                # 1. YOLO PERSON DETECTION
                # -------------------------------------

                results = model.predict(
                    source=frame,
                    classes=[0],
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    verbose=False
                )

                boxes = results[0].boxes

                detections = []

                if boxes is not None and len(boxes) > 0:

                    detections = (
                        boxes.xyxy.cpu()
                        .numpy()
                        .tolist()
                    )

                total_detections += len(detections)

                # -------------------------------------
                # 2. APPEARANCE-BASED RE-ID TRACKING
                # -------------------------------------

                tracks = tracker.update(
                    detections,
                    frame
                )

                total_track_observations += len(tracks)

                # -------------------------------------
                # 2A. LOG EVERY VISIBLE TRACK POSITION
                # -------------------------------------

                for track_id, box in tracks:
                    x1, y1, x2, y2 = map(float, box)
                    trajectory_writer.writerow([
                        frame_number,
                        round(timestamp, 3),
                        track_id,
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ])

                # -------------------------------------
                # 3. LOG TRACKING EVENTS
                # -------------------------------------

                for event in tracker.events:

                    event_type = event["event"]

                    event_counts[event_type] = (
                        event_counts.get(event_type, 0) + 1
                    )

                    details = "; ".join(
                        f"{key}={value}"
                        for key, value in event.items()
                        if key not in (
                            "event",
                            "track_id"
                        )
                    )

                    csv_writer.writerow([
                        frame_number,
                        round(timestamp, 3),
                        event_type,
                        event.get("track_id", ""),
                        details
                    ])

                # -------------------------------------
                # 4. DRAW TRACKING RESULTS
                # -------------------------------------

                for track_id, box in tracks:

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                # -------------------------------------
                # 5. DISPLAY FRAME INFORMATION
                # -------------------------------------

                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Active Tracks: {len(tracker.tracks)}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                writer.write(frame)

                if frame_number % 50 == 0:

                    print(
                        f"Processed {frame_number}/"
                        f"{frames_to_process} frames | "
                        f"Active tracks: {len(tracker.tracks)}"
                    )

    finally:

        cap.release()
        writer.release()
        trajectory_file.close()

    # =================================================
    # GENERATE SUMMARY REPORT
    # =================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as summary_file:

        summary_file.write(
            "APPEARANCE-BASED RE-ID TRACKING SUMMARY\n"
        )

        summary_file.write("=" * 45 + "\n\n")

        summary_file.write(
            f"Input video: {VIDEO_PATH}\n"
        )

        summary_file.write(
            f"FPS: {fps}\n"
        )

        summary_file.write(
            f"Resolution: {width}x{height}\n"
        )

        summary_file.write(
            f"Total video frames: {total_frames}\n"
        )

        summary_file.write(
            f"Maximum frames configured: {MAX_FRAMES}\n"
        )

        summary_file.write(
            f"Total frames processed: {frame_number}\n"
        )

        summary_file.write(
            f"Unique IDs created: {tracker.next_id - 1}\n"
        )

        summary_file.write(
            f"Total detections: {total_detections}\n"
        )

        summary_file.write(
            f"Total track observations: "
            f"{total_track_observations}\n"
        )

        summary_file.write(
            f"Trajectory log: {TRAJECTORY_LOG_PATH}\n"
        )

        summary_file.write("\nTRACKING EVENTS\n")

        for event_name, count in event_counts.items():

            summary_file.write(
                f"{event_name}: {count}\n"
            )

        summary_file.write("\nTRACKER PARAMETERS\n")

        summary_file.write(
            f"Max missed frames: {MAX_MISSED}\n"
        )

        summary_file.write(
            f"Appearance threshold: {APPEARANCE_THRESHOLD}\n"
        )

        summary_file.write(
            f"Max center distance: {MAX_CENTER_DISTANCE}\n"
        )

    # =================================================
    # FINAL OUTPUT
    # =================================================

    print("\nRe-ID Processing Completed!")

    print(f"Frames processed: {frame_number}")

    print(
        f"Output video: {OUTPUT_PATH.resolve()}"
    )

    print(
        f"Event CSV: {EVENT_LOG_PATH.resolve()}"
    )

    print(
        f"Trajectory CSV: {TRAJECTORY_LOG_PATH.resolve()}"
    )

    print(
        f"Summary: {SUMMARY_PATH.resolve()}"
    )

    print("\nTracking Event Counts:")

    for event_name, count in event_counts.items():

        print(
            f"{event_name}: {count}"
        )


if __name__ == "__main__":
    main()
