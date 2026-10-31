"""
Final High-Accuracy Hybrid Multi-Object Tracker.

Integrates:
1. BoT-SORT Direct (cx, cy, w, h, vx, vy, vw, vh) Kalman Filter
2. Two-Stage BYTE Association (D_high -> D_low)
3. Deep OSNet Appearance Embedding Extraction
4. Multi-Sample Feature Gallery with Occlusion Quality Gating
5. Hard Spatial-Motion Gating before Hungarian Assignment
6. Track Confirmation Hysteresis (anti-false-positive track initiation)
"""

from typing import List, Tuple, Optional, Union
import numpy as np

from src.detection.detector import Detection
from src.detection.postprocessing import xywh_to_xyxy
from src.tracking.track import Track, TrackState
from src.tracking.track_manager import TrackManager
from src.tracking.association import iou_distance, fused_cost_matrix, box_ious
from src.tracking.hungarian_matcher import linear_assignment
from src.reid.feature_extractor import OSNetFeatureExtractor
from src.reid.reidentifier import compute_appearance_distance_matrix


class HybridReIDTracker:
    """
    State-of-the-art hybrid multi-object tracker optimizing MOTA and IDF1.
    """

    def __init__(
        self,
        conf_high: float = 0.50,
        conf_low: float = 0.10,
        match_thresh_stage1: float = 0.80,
        match_thresh_stage2: float = 0.50,
        iou_weight: float = 0.40,
        reid_weight: float = 0.60,
        spatial_gate_dist: float = 300.0,
        track_buffer: int = 90,
        confirm_window: int = 2,
        feature_extractor: Optional[OSNetFeatureExtractor] = None
    ):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.iou_weight = iou_weight
        self.reid_weight = reid_weight
        self.spatial_gate_dist = spatial_gate_dist

        self.feature_extractor = feature_extractor
        self.manager = TrackManager(track_buffer=track_buffer, confirm_window=confirm_window)
        self.frame_id = 0

    def reset(self):
        self.manager.reset()
        self.frame_id = 0

    def _extract_appearance(
        self,
        frame: Optional[np.ndarray],
        boxes: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """Extract OSNet feature embeddings if extractor and frame are available."""
        if self.feature_extractor is None or frame is None or len(boxes) == 0:
            return [None] * len(boxes)
        return self.feature_extractor.extract_batch(frame, boxes)

    def update(
        self,
        detections: Union[List[Detection], List[Tuple[float, float, float, float]], np.ndarray],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Advance tracker state by one frame.

        Args:
            detections: List of Detection objects, or raw [x1, y1, x2, y2] boxes.
            frame: Optional BGR image frame for OSNet appearance extraction.
        Returns:
            List of confirmed active Track objects.
        """
        self.frame_id += 1

        # Normalize detections input to List[Detection]
        norm_dets: List[Detection] = []
        if isinstance(detections, np.ndarray):
            for row in detections:
                if len(row) >= 5:
                    norm_dets.append(Detection(box=row[:4], score=float(row[4])))
                elif len(row) == 4:
                    norm_dets.append(Detection(box=row[:4], score=1.0))
        elif isinstance(detections, list):
            for d in detections:
                if isinstance(d, Detection):
                    norm_dets.append(d)
                elif isinstance(d, (list, tuple)) and len(d) >= 4:
                    score = float(d[4]) if len(d) > 4 else 1.0
                    norm_dets.append(Detection(box=np.array(d[:4], dtype=float), score=score))

        # 1. Kalman prediction for all active tracks
        self.manager.predict_all()

        # 2. Partition detections into high-score (Stage 1) and low-score (Stage 2)
        d_high: List[Detection] = []
        d_low: List[Detection] = []
        for d in norm_dets:
            if d.score >= self.conf_high:
                d_high.append(d)
            elif d.score >= self.conf_low:
                d_low.append(d)

        # Compute mutual overlap between detections to prevent updating gallery on heavily overlapping boxes
        high_boxes = np.array([d.box for d in d_high], dtype=float) if d_high else np.empty((0, 4))
        high_overlaps = np.zeros(len(d_high), dtype=float)
        if len(d_high) > 1:
            ious_all = box_ious(high_boxes, high_boxes)
            np.fill_diagonal(ious_all, 0.0)
            high_overlaps = np.max(ious_all, axis=1)

        # Extract appearance features for high-confidence detections
        high_features = self._extract_appearance(frame, [d.box for d in d_high])

        # ------------------------------------------------------------------
        # Stage 1: Associate active tracks with high-confidence detections
        # ------------------------------------------------------------------
        active_tracks = self.manager.all_active_tracks
        track_boxes = np.array([t.box for t in active_tracks], dtype=float) if active_tracks else np.empty((0, 4))

        if active_tracks and d_high:
            # Build appearance distance matrix
            galleries = [t.gallery for t in active_tracks]
            app_dist_mat = compute_appearance_distance_matrix(galleries, high_features, max_distance=1.0)

            # Build fused cost matrix with spatial gating
            cost_stage1 = fused_cost_matrix(
                boxes_a=track_boxes,
                boxes_b=high_boxes,
                appearance_distance=app_dist_mat,
                alpha=self.iou_weight,
                spatial_gate_dist=self.spatial_gate_dist
            )
        else:
            cost_stage1 = np.empty((len(active_tracks), len(d_high)), dtype=float)

        matches1, unmatched_tracks1_idx, unmatched_d_high_idx = linear_assignment(
            cost_stage1, threshold=self.match_thresh_stage1
        )

        for track_idx, det_idx in matches1:
            track = active_tracks[track_idx]
            det = d_high[det_idx]
            feat = high_features[det_idx]
            ov = float(high_overlaps[det_idx]) if len(high_overlaps) > det_idx else 0.0
            track.update(box=det.box, score=det.score, feature=feat, frame_id=self.frame_id, max_overlap_with_others=ov)
            self.manager.update_confirmed(track)

        # ------------------------------------------------------------------
        # Stage 2: Associate remaining tracks with low-confidence detections
        # ------------------------------------------------------------------
        unmatched_tracks1 = [active_tracks[i] for i in unmatched_tracks1_idx]
        rem_track_boxes = np.array([t.box for t in unmatched_tracks1], dtype=float) if unmatched_tracks1 else np.empty((0, 4))
        low_boxes = np.array([d.box for d in d_low], dtype=float) if d_low else np.empty((0, 4))

        cost_stage2 = iou_distance(rem_track_boxes, low_boxes)
        matches2, _, _ = linear_assignment(cost_stage2, threshold=self.match_thresh_stage2)

        for track_idx, det_idx in matches2:
            track = unmatched_tracks1[track_idx]
            det = d_low[det_idx]
            # Low confidence detections: update position without updating appearance gallery
            track.update(box=det.box, score=det.score, feature=None, frame_id=self.frame_id)
            self.manager.update_confirmed(track)

        # ------------------------------------------------------------------
        # Stage 3: Match candidate tracks and initialize new tracks
        # ------------------------------------------------------------------
        unmatched_d_high = [d_high[i] for i in unmatched_d_high_idx]
        unmatched_high_features = [high_features[i] for i in unmatched_d_high_idx]
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
                feat = unmatched_high_features[det_idx]
                track.update(box=det.box, score=det.score, feature=feat, frame_id=self.frame_id)
                matched_candidates.append(track)
            final_d_high = [unmatched_d_high[i] for i in final_unmatched_d_high_idx]
            final_d_feats = [unmatched_high_features[i] for i in final_unmatched_d_high_idx]
        else:
            final_d_high = unmatched_d_high
            final_d_feats = unmatched_high_features

        for track in matched_candidates:
            self.manager.update_confirmed(track)

        # Initialize new candidates from remaining high-confidence detections
        for det, feat in zip(final_d_high, final_d_feats):
            new_track = Track(box=det.box, score=det.score, feature=feat)
            self.manager.add_new_candidate(new_track)

        # 4. Advance lifecycle (confirmations, timeouts, expirations)
        self.manager.step_lifecycle()

        return self.manager.get_output_tracks()
