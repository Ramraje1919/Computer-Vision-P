
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# =========================================================
# IOU CALCULATION
# =========================================================

def iou(box_a, box_b):
    """Calculate Intersection over Union for xyxy boxes."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# =========================================================
# BOX CONVERSION
# =========================================================

def box_to_measurement(box):
    """Convert xyxy to center-x, center-y, width, height."""

    x1, y1, x2, y2 = box

    return np.array([
        [(x1 + x2) / 2],
        [(y1 + y2) / 2],
        [max(1.0, x2 - x1)],
        [max(1.0, y2 - y1)]
    ], dtype=np.float32)


def measurement_to_box(measurement):
    """Convert Kalman state to xyxy box."""

    cx, cy, w, h = measurement.flatten()

    w = max(1.0, w)
    h = max(1.0, h)

    return np.array([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ], dtype=float)


# =========================================================
# TRACK CLASS
# =========================================================

class Track:

    def __init__(self, track_id, box):

        self.track_id = track_id
        self.box = np.array(box, dtype=float)

        self.missed_frames = 0
        self.hits = 1
        self.age = 1

        # Kalman state:
        # cx, cy, width, height,
        # vx, vy, vw, vh

        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.transitionMatrix = np.eye(
            8, dtype=np.float32
        )

        self.kf.transitionMatrix[0, 4] = 1
        self.kf.transitionMatrix[1, 5] = 1
        self.kf.transitionMatrix[2, 6] = 1
        self.kf.transitionMatrix[3, 7] = 1

        self.kf.measurementMatrix = np.zeros(
            (4, 8), dtype=np.float32
        )

        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1
        self.kf.measurementMatrix[3, 3] = 1

        self.kf.processNoiseCov = (
            np.eye(8, dtype=np.float32) * 0.03
        )

        self.kf.measurementNoiseCov = (
            np.eye(4, dtype=np.float32) * 1.0
        )

        self.kf.errorCovPost = (
            np.eye(8, dtype=np.float32) * 10
        )

        self.kf.statePost = np.zeros(
            (8, 1), dtype=np.float32
        )

        self.kf.statePost[:4] = box_to_measurement(box)

    def predict(self):

        prediction = self.kf.predict()

        self.box = measurement_to_box(
            prediction[:4]
        )

        self.age += 1

        return self.box

    def update(self, box):

        measurement = box_to_measurement(box)

        self.kf.correct(measurement)

        self.box = np.array(box, dtype=float)

        self.missed_frames = 0
        self.hits += 1


# =========================================================
# MULTI OBJECT TRACKER
# =========================================================

class IoUTracker:

    """
    Kalman + Hungarian + Lost Track Re-association.

    Events:
    NEW_TRACK
    TRACK_MISSING
    TRACK_RECOVERED
    TRACK_EXPIRED

    Note:
    Motion-based recovery only.
    This is not appearance-based Re-ID.
    """

    def __init__(
        self,
        iou_threshold=0.10,
        max_missed=90,
        max_center_distance=150,
        recovery_distance=200
    ):

        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_distance = max_center_distance
        self.recovery_distance = recovery_distance

        self.next_id = 1

        # All active and temporarily lost tracks.
        self.tracks = {}

        self.events = []

    # -----------------------------------------------------
    # MATCHING COST
    # -----------------------------------------------------

    def calculate_cost(self, predicted_box, detection):

        pcx = (predicted_box[0] + predicted_box[2]) / 2
        pcy = (predicted_box[1] + predicted_box[3]) / 2

        dcx = (detection[0] + detection[2]) / 2
        dcy = (detection[1] + detection[3]) / 2

        distance = np.hypot(
            pcx - dcx,
            pcy - dcy
        )

        overlap = iou(predicted_box, detection)

        if (
            distance > self.max_center_distance
            or (
                overlap < self.iou_threshold
                and distance > 30
            )
        ):
            return 1e6

        normalized_distance = min(
            distance / self.max_center_distance,
            1.0
        )

        return (
            0.6 * (1 - overlap)
            + 0.4 * normalized_distance
        )

    # -----------------------------------------------------
    # UPDATE TRACKER
    # -----------------------------------------------------

    def update(self, detections):

        self.events = []

        detections = [
            np.array(box, dtype=float)
            for box in detections
        ]

        track_ids = list(self.tracks.keys())

        predictions = {}

        # Predict all existing tracks.
        for track_id in track_ids:

            predictions[track_id] = (
                self.tracks[track_id].predict()
            )

        matched_tracks = set()
        matched_detections = set()

        # -------------------------------------------------
        # FIRST PASS: NORMAL MATCHING
        # -------------------------------------------------

        if track_ids and detections:

            cost_matrix = np.full(
                (len(track_ids), len(detections)),
                1e6,
                dtype=float
            )

            for i, track_id in enumerate(track_ids):

                predicted_box = predictions[track_id]

                for j, detection in enumerate(detections):

                    cost_matrix[i, j] = self.calculate_cost(
                        predicted_box,
                        detection
                    )

            rows, cols = linear_sum_assignment(
                cost_matrix
            )

            for row, col in zip(rows, cols):

                if cost_matrix[row, col] >= 1e5:
                    continue

                track_id = track_ids[row]

                previous_missed = (
                    self.tracks[track_id].missed_frames
                )

                self.tracks[track_id].update(
                    detections[col]
                )

                matched_tracks.add(track_id)
                matched_detections.add(col)

                if previous_missed > 0:

                    self.events.append({
                        "event": "TRACK_RECOVERED",
                        "track_id": track_id,
                        "missed_frames": previous_missed
                    })

        # -------------------------------------------------
        # SECOND PASS: LOST TRACK RE-ASSOCIATION
        # -------------------------------------------------

        unmatched_detections = [
            index
            for index in range(len(detections))
            if index not in matched_detections
        ]

        unmatched_lost_tracks = [
            track_id
            for track_id in track_ids
            if (
                track_id not in matched_tracks
                and self.tracks[track_id].missed_frames > 0
            )
        ]

        if unmatched_lost_tracks and unmatched_detections:

            recovery_cost = np.full(
                (
                    len(unmatched_lost_tracks),
                    len(unmatched_detections)
                ),
                1e6,
                dtype=float
            )

            for i, track_id in enumerate(unmatched_lost_tracks):

                predicted_box = predictions[track_id]

                for j, det_index in enumerate(unmatched_detections):

                    detection = detections[det_index]

                    pcx = (
                        predicted_box[0] + predicted_box[2]
                    ) / 2

                    pcy = (
                        predicted_box[1] + predicted_box[3]
                    ) / 2

                    dcx = (
                        detection[0] + detection[2]
                    ) / 2

                    dcy = (
                        detection[1] + detection[3]
                    ) / 2

                    distance = np.hypot(
                        pcx - dcx,
                        pcy - dcy
                    )

                    overlap = iou(
                        predicted_box,
                        detection
                    )

                    if (
                        distance <= self.recovery_distance
                        and (
                            overlap >= 0.01
                            or distance <= 50
                        )
                    ):

                        recovery_cost[i, j] = (
                            0.7 * min(
                                distance / self.recovery_distance,
                                1.0
                            )
                            + 0.3 * (1 - overlap)
                        )

            rows, cols = linear_sum_assignment(
                recovery_cost
            )

            for row, col in zip(rows, cols):

                if recovery_cost[row, col] >= 1e5:
                    continue

                track_id = unmatched_lost_tracks[row]
                det_index = unmatched_detections[col]

                previous_missed = (
                    self.tracks[track_id].missed_frames
                )

                self.tracks[track_id].update(
                    detections[det_index]
                )

                matched_tracks.add(track_id)
                matched_detections.add(det_index)

                self.events.append({
                    "event": "TRACK_RECOVERED",
                    "track_id": track_id,
                    "missed_frames": previous_missed,
                    "method": "LOST_TRACK_REASSOCIATION"
                })

        # -------------------------------------------------
        # RECORD MISSING TRACKS
        # -------------------------------------------------

        for track_id in track_ids:

            if track_id not in matched_tracks:

                self.tracks[track_id].missed_frames += 1

                self.events.append({
                    "event": "TRACK_MISSING",
                    "track_id": track_id,
                    "missed_frames": (
                        self.tracks[track_id].missed_frames
                    )
                })

        # -------------------------------------------------
        # CREATE NEW TRACKS
        # -------------------------------------------------

        for index, detection in enumerate(detections):

            if index not in matched_detections:

                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = Track(
                    track_id,
                    detection
                )

                matched_tracks.add(track_id)

                self.events.append({
                    "event": "NEW_TRACK",
                    "track_id": track_id,
                    "detection_index": index
                })

        # -------------------------------------------------
        # REMOVE EXPIRED TRACKS
        # -------------------------------------------------

        expired_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed
        ]

        for track_id in expired_tracks:

            self.events.append({
                "event": "TRACK_EXPIRED",
                "track_id": track_id,
                "missed_frames": (
                    self.tracks[track_id].missed_frames
                )
            })

            del self.tracks[track_id]

        # -------------------------------------------------
        # RETURN CURRENTLY DETECTED TRACKS
        # -------------------------------------------------

        return [
            (track_id, track.box.copy())
            for track_id, track in self.tracks.items()
            if track.missed_frames == 0
        ]
