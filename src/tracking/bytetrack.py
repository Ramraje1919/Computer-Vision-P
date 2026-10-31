"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

Reference:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
"""

from typing import List, Tuple, Optional
import numpy as np

from src.detection.detector import Detection
from src.tracking.track import Track
from src.tracking.track_manager import TrackManager
from src.tracking.association import iou_distance
from src.tracking.hungarian_matcher import linear_assignment


class ByteTracker:
    """
    Two-stage BYTE association tracker.
    Associates high-confidence detections first, then associates low-confidence
    detections to recover occluded tracks without creating false positive tracks.
    """

    def __init__(
        self,
        conf_high: float = 0.50,
        conf_low: float = 0.10,
        match_thresh_stage1: float = 0.80,
        match_thresh_stage2: float = 0.50,
        track_buffer: int = 90,
        confirm_window: int = 2
    ):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2

        self.manager = TrackManager(track_buffer=track_buffer, confirm_window=confirm_window)
        self.frame_id = 0

    def reset(self):
        self.manager.reset()
        self.frame_id = 0

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Run ByteTrack update for a single frame.
        Returns: List of confirmed active tracks.
        """
        self.frame_id += 1

        # 1. Kalman state prediction for all tracks
        self.manager.predict_all()

        # 2. Partition detections into high-score and low-score subsets
        d_high: List[Detection] = []
        d_low: List[Detection] = []

        for d in detections:
            if d.score >= self.conf_high:
                d_high.append(d)
            elif d.score >= self.conf_low:
                d_low.append(d)

        # ------------------------------------------------------------------
        # Stage 1: Associate active tracks with high-confidence detections
        # ------------------------------------------------------------------
        active_tracks = self.manager.all_active_tracks
        track_boxes = np.array([t.box for t in active_tracks], dtype=float) if active_tracks else np.empty((0, 4))
        high_boxes = np.array([d.box for d in d_high], dtype=float) if d_high else np.empty((0, 4))

        cost_stage1 = iou_distance(track_boxes, high_boxes)
        matches1, unmatched_tracks1_idx, unmatched_d_high_idx = linear_assignment(
            cost_stage1, threshold=self.match_thresh_stage1
        )

        for track_idx, det_idx in matches1:
            track = active_tracks[track_idx]
            det = d_high[det_idx]
            track.update(box=det.box, score=det.score, frame_id=self.frame_id)
            self.manager.update_confirmed(track)

        # ------------------------------------------------------------------
        # Stage 2: Associate remaining active tracks with low-confidence boxes
        # ------------------------------------------------------------------
        unmatched_tracks1 = [active_tracks[i] for i in unmatched_tracks1_idx]
        rem_track_boxes = np.array([t.box for t in unmatched_tracks1], dtype=float) if unmatched_tracks1 else np.empty((0, 4))
        low_boxes = np.array([d.box for d in d_low], dtype=float) if d_low else np.empty((0, 4))

        cost_stage2 = iou_distance(rem_track_boxes, low_boxes)
        matches2, unmatched_tracks2_idx, _ = linear_assignment(
            cost_stage2, threshold=self.match_thresh_stage2
        )

        for track_idx, det_idx in matches2:
            track = unmatched_tracks1[track_idx]
            det = d_low[det_idx]
            track.update(box=det.box, score=det.score, frame_id=self.frame_id)
            self.manager.update_confirmed(track)

        # ------------------------------------------------------------------
        # Stage 3: Associate new candidates with unmatched high-score boxes
        # ------------------------------------------------------------------
        unmatched_d_high = [d_high[i] for i in unmatched_d_high_idx]
        candidate_tracks = list(self.manager.new_tracks)

        matched_candidates = []
        if candidate_tracks and unmatched_d_high:
            cand_boxes = np.array([t.box for t in candidate_tracks], dtype=float)
            unmatched_high_boxes = np.array([d.box for d in unmatched_d_high], dtype=float)
            cost_stage3 = iou_distance(cand_boxes, unmatched_high_boxes)
            matches3, _, final_unmatched_d_high_idx = linear_assignment(
                cost_stage3, threshold=self.match_thresh_stage1
            )
            for cand_idx, det_idx in matches3:
                track = candidate_tracks[cand_idx]
                det = unmatched_d_high[det_idx]
                track.update(box=det.box, score=det.score, frame_id=self.frame_id)
                matched_candidates.append(track)
            final_d_high = [unmatched_d_high[i] for i in final_unmatched_d_high_idx]
        else:
            final_d_high = unmatched_d_high

        for track in matched_candidates:
            self.manager.update_confirmed(track)

        # Initialize new track candidates from remaining high-confidence detections
        for det in final_d_high:
            new_track = Track(box=det.box, score=det.score)
            self.manager.add_new_candidate(new_track)

        # 4. Advance track lifecycle (confirmations, timeouts, expirations)
        self.manager.step_lifecycle()

        return self.manager.get_output_tracks()
