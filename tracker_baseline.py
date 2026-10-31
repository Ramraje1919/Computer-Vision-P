
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b):
    """Calculate Intersection over Union for two xyxy boxes."""
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


class Track:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = np.array(box, dtype=float)
        self.missed_frames = 0


class IoUTracker:
    """
    Baseline tracker using IoU and Hungarian matching.
    Kalman prediction and appearance-based re-identification
    will be evaluated as subsequent improvements.
    """

    def __init__(self, iou_threshold=0.20, max_missed=15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2] boxes.
        Returns: list of (track_id, box).
        """
        detections = [
            np.array(box, dtype=float) for box in detections
        ]

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]

        matched_tracks = set()
        matched_detections = set()

        if track_boxes and detections:
            cost = np.zeros(
                (len(track_boxes), len(detections)),
                dtype=float
            )

            for i, track_box in enumerate(track_boxes):
                for j, det_box in enumerate(detections):
                    cost[i, j] = 1.0 - iou(track_box, det_box)

            rows, cols = linear_sum_assignment(cost)

            for row, col in zip(rows, cols):
                overlap = 1.0 - cost[row, col]

                if overlap >= self.iou_threshold:
                    track_id = track_ids[row]
                    self.tracks[track_id].box = detections[col]
                    self.tracks[track_id].missed_frames = 0
                    matched_tracks.add(track_id)
                    matched_detections.add(col)

        # Keep unmatched tracks temporarily for possible recovery.
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].missed_frames += 1

        # Create tracks for unmatched detections.
        for index, box in enumerate(detections):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id, box)

        # Remove tracks that have been missing too long.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return [
            (track_id, track.box.copy())
            for track_id, track in self.tracks.items()
            if track.missed_frames == 0
        ]
