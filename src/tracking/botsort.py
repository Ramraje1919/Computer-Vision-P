"""
BoT-SORT: Robust Multi-Person Tracking by Integrating Motion and Appearance Information.

Reference:
    Aharon et al., "BoT-SORT: Robust Multi-Person Tracking by Integrating Motion and Appearance Information", arXiv 2022.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.detection.detector import Detection
from src.tracking.track import Track
from src.tracking.track_manager import TrackManager
from src.tracking.association import iou_distance, fused_cost_matrix
from src.tracking.hungarian_matcher import linear_assignment


class BoTSORTTracker:
    """
    BoT-SORT tracker featuring:
    - 8-state direct (cx, cy, w, h, vx, vy, vw, vh) Kalman filter
    - Camera Motion Compensation (CMC) support
    - Two-stage BYTE association
    """

    def __init__(
        self,
        conf_high: float = 0.50,
        conf_low: float = 0.10,
        match_thresh_stage1: float = 0.80,
        match_thresh_stage2: float = 0.50,
        track_buffer: int = 90,
        confirm_window: int = 2,
        enable_cmc: bool = False
    ):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.enable_cmc = enable_cmc

        self.manager = TrackManager(track_buffer=track_buffer, confirm_window=confirm_window)
        self.frame_id = 0
        self.prev_frame_gray: Optional[np.ndarray] = None

    def reset(self):
        self.manager.reset()
        self.frame_id = 0
        self.prev_frame_gray = None

    def _estimate_camera_motion(self, frame_gray: np.ndarray) -> np.ndarray:
        """Estimate 2x3 affine camera transformation using ORB features."""
        if self.prev_frame_gray is None:
            return np.eye(2, 3, dtype=np.float32)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(self.prev_frame_gray, None)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return np.eye(2, 3, dtype=np.float32)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 8:
            return np.eye(2, 3, dtype=np.float32)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        warp_matrix, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
        if warp_matrix is None:
            return np.eye(2, 3, dtype=np.float32)
        return warp_matrix

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        self.frame_id += 1

        # Camera motion compensation
        if self.enable_cmc and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            warp = self._estimate_camera_motion(gray)
            self.prev_frame_gray = gray
            # Compensate Kalman positions
            R = warp[:2, :2]
            T = warp[:2, 2]
            for track in self.manager.all_active_tracks:
                pos = track.mean[:2]
                pos_comp = np.dot(R, pos) + T
                track.mean[0] = pos_comp[0]
                track.mean[1] = pos_comp[1]

        # 1. Kalman prediction
        self.manager.predict_all()

        # 2. Split detections
        d_high: List[Detection] = []
        d_low: List[Detection] = []
        for d in detections:
            if d.score >= self.conf_high:
                d_high.append(d)
            elif d.score >= self.conf_low:
                d_low.append(d)

        # 3. Stage 1: High-score matching
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

        # 4. Stage 2: Low-score matching
        unmatched_tracks1 = [active_tracks[i] for i in unmatched_tracks1_idx]
        rem_track_boxes = np.array([t.box for t in unmatched_tracks1], dtype=float) if unmatched_tracks1 else np.empty((0, 4))
        low_boxes = np.array([d.box for d in d_low], dtype=float) if d_low else np.empty((0, 4))

        cost_stage2 = iou_distance(rem_track_boxes, low_boxes)
        matches2, _, _ = linear_assignment(cost_stage2, threshold=self.match_thresh_stage2)

        for track_idx, det_idx in matches2:
            track = unmatched_tracks1[track_idx]
            det = d_low[det_idx]
            track.update(box=det.box, score=det.score, frame_id=self.frame_id)
            self.manager.update_confirmed(track)

        # 5. Candidate track matching and new track initialization
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

        for det in final_d_high:
            new_track = Track(box=det.box, score=det.score)
            self.manager.add_new_candidate(new_track)

        self.manager.step_lifecycle()
        return self.manager.get_output_tracks()
