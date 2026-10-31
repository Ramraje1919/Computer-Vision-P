"""
Track Representation and Finite State Machine Entity.
"""

from enum import IntEnum
from typing import List, Tuple, Optional
import numpy as np

from src.detection.postprocessing import xyxy_to_cxcywh, cxcywh_to_xyxy
from src.tracking.kalman_filter import KalmanFilter
from src.reid.feature_gallery import FeatureGallery


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    EXPIRED = 3


class Track:
    """
    Core stateful track object.
    Maintains Kalman state, appearance gallery, trajectory trail, and lifecycle state.
    """

    _count = 0

    def __init__(
        self,
        box: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        max_gallery_samples: int = 6,
        kalman_filter: Optional[KalmanFilter] = None
    ):
        Track._count += 1
        self.track_id = Track._count
        self.state = TrackState.NEW
        self.score = float(score)

        # Initial bounding box [x1, y1, x2, y2]
        self.box = np.asarray(box, dtype=float).copy()

        # Kalman Filter initialization
        self.kf = kalman_filter if kalman_filter is not None else KalmanFilter()
        measurement = xyxy_to_cxcywh(self.box)
        self.mean, self.covariance = self.kf.initiate(measurement)

        # Appearance gallery
        self.gallery = FeatureGallery(max_samples=max_gallery_samples)
        if feature is not None:
            self.gallery.add_initial_feature(feature)

        self.age = 1
        self.hits = 1
        self.missed_frames = 0
        self.frame_id = 0

        # Trajectory history: list of (frame_id, cx, cy)
        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.trajectory: List[Tuple[int, float, float]] = [(0, cx, cy)]

    @classmethod
    def reset_counter(cls, start_id: int = 0):
        cls._count = start_id

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.TRACKED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_expired(self) -> bool:
        return self.state == TrackState.EXPIRED

    @property
    def center(self) -> Tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0, float(self.box[1] + self.box[3]) / 2.0)

    def predict(self):
        """Advance Kalman state forward by one frame."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        # Update current box from projected position
        measurement = self.mean[:4]
        self.box = cxcywh_to_xyxy(measurement)
        self.age += 1
        self.missed_frames += 1

    def update(
        self,
        box: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        frame_id: int = 0,
        max_overlap_with_others: float = 0.0
    ):
        """Update track with an associated detection observation."""
        self.box = np.asarray(box, dtype=float).copy()
        self.score = float(score)
        self.frame_id = frame_id
        measurement = xyxy_to_cxcywh(self.box)

        # Correct Kalman Filter state
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)

        # Update appearance gallery with quality gating
        if feature is not None:
            self.gallery.update(feature, score=score, max_overlap_with_others=max_overlap_with_others)

        self.hits += 1
        self.missed_frames = 0
        self.state = TrackState.TRACKED

        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.trajectory.append((frame_id, cx, cy))

    def mark_lost(self):
        """Mark as lost (temporarily missing, search in recovery buffer)."""
        self.state = TrackState.LOST

    def mark_expired(self):
        """Mark as expired (removed from active tracking pool)."""
        self.state = TrackState.EXPIRED

