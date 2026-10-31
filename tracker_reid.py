import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .appearance_model import OSNetAppearance
except ImportError:
    from appearance_model import OSNetAppearance


def iou(box_a, box_b):
    """Calculate intersection over union for two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def normalize_feature(feature):
    """Return an L2-normalized OSNet embedding."""
    if feature is None:
        return None

    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))

    if feature.size == 0 or norm < 1e-12:
        return None

    return feature / norm


def appearance_distance(feature_a, feature_b):
    """Cosine distance for OSNet embeddings (0 means most similar)."""
    if feature_a is None or feature_b is None:
        return 1.0

    a = normalize_feature(feature_a)
    b = normalize_feature(feature_b)

    if a is None or b is None or a.shape != b.shape:
        return 1.0

    similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - similarity


def box_to_measurement(box):
    x1, y1, x2, y2 = box

    return np.array(
        [
            [(x1 + x2) / 2.0],
            [(y1 + y2) / 2.0],
            [max(1.0, x2 - x1)],
            [max(1.0, y2 - y1)],
        ],
        dtype=np.float32,
    )


def measurement_to_box(measurement):
    cx, cy, width, height = measurement.flatten()
    width = max(1.0, float(width))
    height = max(1.0, float(height))

    return np.array(
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ],
        dtype=float,
    )


class Track:
    def __init__(self, track_id, box, appearance):
        self.track_id = track_id
        self.box = np.asarray(box, dtype=float).copy()
        self.appearance = normalize_feature(appearance)
        self.missed_frames = 0

        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0
        self.kf.transitionMatrix[1, 5] = 1.0
        self.kf.transitionMatrix[2, 6] = 1.0
        self.kf.transitionMatrix[3, 7] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = box_to_measurement(box)

    def predict(self):
        prediction = self.kf.predict()
        self.box = measurement_to_box(prediction[:4])
        return self.box

    def update(self, box, appearance, appearance_momentum=0.90):
        self.kf.correct(box_to_measurement(box))
        self.box = np.asarray(box, dtype=float).copy()
        self.missed_frames = 0

        new_feature = normalize_feature(appearance)
        if new_feature is not None:
            if self.appearance is None:
                self.appearance = new_feature
            else:
                blended = (
                    appearance_momentum * self.appearance
                    + (1.0 - appearance_momentum) * new_feature
                )
                self.appearance = normalize_feature(blended)


class IoUTracker:
    """
    Custom MOT tracker using Kalman prediction, IoU, OSNet Re-ID,
    Hungarian assignment, and temporary track recovery.

    The public interface remains IoUTracker.update(detections, frame).
    This implementation does not use ByteTrack.
    """

    def __init__(
        self,
        iou_threshold=0.10,
        max_missed=150,
        max_center_distance=350,
        appearance_threshold=0.35,
        appearance_momentum=0.90,
        appearance_extractor=None,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.max_center_distance = float(max_center_distance)
        self.appearance_threshold = float(appearance_threshold)
        self.appearance_momentum = float(appearance_momentum)

        # Load the OSNet model once, not once per frame.
        self.appearance_extractor = (
            appearance_extractor
            if appearance_extractor is not None
            else OSNetAppearance()
        )

        self.next_id = 1
        self.tracks = {}
        self.events = []

    def _extract_features(self, detections, frame):
        features = []
        for box in detections:
            feature = self.appearance_extractor.extract(frame, box)
            features.append(normalize_feature(feature))
        return features

    def update(self, detections, frame):
        self.events = []
        detections = [
            np.asarray(box, dtype=float).reshape(4)
            for box in detections
        ]

        features = self._extract_features(detections, frame)

        track_ids = list(self.tracks.keys())
        predictions = {}

        # 1. Predict the position of every existing track.
        for track_id in track_ids:
            predictions[track_id] = self.tracks[track_id].predict()

        matched_tracks = set()
        matched_detections = set()

        # 2. Build the track-to-detection cost matrix.
        if track_ids and detections:
            invalid_cost = 1e6
            cost = np.full(
                (len(track_ids), len(detections)),
                invalid_cost,
                dtype=np.float64,
            )

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted_box = predictions[track_id]
                pcx = (predicted_box[0] + predicted_box[2]) / 2.0
                pcy = (predicted_box[1] + predicted_box[3]) / 2.0

                for j, det_box in enumerate(detections):
                    dcx = (det_box[0] + det_box[2]) / 2.0
                    dcy = (det_box[1] + det_box[3]) / 2.0
                    distance = float(np.hypot(pcx - dcx, pcy - dcy))
                    overlap = iou(predicted_box, det_box)
                    app_distance = appearance_distance(
                        track.appearance, features[j]
                    )

                    spatial_match = (
                        distance <= self.max_center_distance
                        and (
                            overlap >= self.iou_threshold
                            or distance <= 50.0
                        )
                    )
                    appearance_match = (
                        track.appearance is not None
                        and features[j] is not None
                        and app_distance <= self.appearance_threshold
                        and distance <= self.max_center_distance
                    )

                    if not (spatial_match or appearance_match):
                        continue

                    normalized_distance = min(
                        distance / max(self.max_center_distance, 1.0),
                        1.0,
                    )

                    # Lower cost means a better association.
                    cost[i, j] = (
                        0.25 * (1.0 - overlap)
                        + 0.15 * normalized_distance
                        + 0.60 * min(app_distance, 2.0) / 2.0
                    )

            # 3. Hungarian assignment.
            rows, cols = linear_sum_assignment(cost)

            for row, col in zip(rows, cols):
                if cost[row, col] >= invalid_cost:
                    continue

                track_id = track_ids[row]
                track = self.tracks[track_id]
                previous_missed = track.missed_frames

                track.update(
                    detections[col],
                    features[col],
                    appearance_momentum=self.appearance_momentum,
                )

                matched_tracks.add(track_id)
                matched_detections.add(col)

                if previous_missed > 0:
                    self.events.append(
                        {
                            "event": "TRACK_RECOVERED",
                            "track_id": track_id,
                            "missed_frames": previous_missed,
                        }
                    )

        # 4. Mark unmatched tracks as missing.
        for track_id in track_ids:
            if track_id not in matched_tracks:
                track = self.tracks[track_id]

                if track.missed_frames == 0:
                    self.events.append(
                        {
                            "event": "TRACK_MISSING",
                            "track_id": track_id,
                            "missed_frames": 1,
                        }
                    )

                track.missed_frames += 1

        # 5. Create tracks for unmatched detections.
        for index, box in enumerate(detections):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = Track(
                    track_id, box, features[index]
                )

                self.events.append(
                    {
                        "event": "NEW_TRACK",
                        "track_id": track_id,
                    }
                )

        # 6. Expire tracks that have been missing for too long.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed
        ]

        for track_id in expired:
            self.events.append(
                {
                    "event": "TRACK_EXPIRED",
                    "track_id": track_id,
                }
            )
            del self.tracks[track_id]

        # 7. Return visible tracks, preserving the previous interface.
        return [
            (track_id, track.box.copy())
            for track_id, track in self.tracks.items()
            if track.missed_frames == 0
        ]
