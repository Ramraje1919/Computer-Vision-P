"""
Unit tests for all core tracking, motion, ReID, counting, and evaluation components.
"""

import numpy as np
import pytest

from src.detection.postprocessing import (
    xyxy_to_xywh, xywh_to_xyxy, xyxy_to_cxcywh, cxcywh_to_xyxy, nms, clip_boxes
)
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.hungarian_matcher import linear_assignment
from src.tracking.association import box_ious, iou_distance, fused_cost_matrix
from src.tracking.track import Track, TrackState
from src.tracking.track_manager import TrackManager
from src.tracking.bytetrack import ByteTracker
from src.tracking.tracker import HybridReIDTracker
from src.detection.detector import Detection
from src.reid.feature_gallery import FeatureGallery
from src.counting.line_counter import DirectionalLineCounter, line_segment_intersection


# 1. Coordinate transformations
def test_box_conversions():
    xyxy = np.array([10.0, 20.0, 50.0, 100.0])
    xywh = xyxy_to_xywh(xyxy)
    assert np.allclose(xywh, [10.0, 20.0, 40.0, 80.0])
    assert np.allclose(xywh_to_xyxy(xywh), xyxy)

    cxcywh = xyxy_to_cxcywh(xyxy)
    assert np.allclose(cxcywh, [30.0, 60.0, 40.0, 80.0])
    assert np.allclose(cxcywh_to_xyxy(cxcywh), xyxy)


# 2. IoU calculation
def test_iou_calculation():
    box1 = np.array([[0, 0, 10, 10]])
    box2 = np.array([[0, 0, 10, 10]])
    assert np.isclose(box_ious(box1, box2)[0, 0], 1.0)

    box3 = np.array([[5, 0, 15, 10]])
    # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 1/3
    assert np.isclose(box_ious(box1, box3)[0, 0], 50.0 / 150.0)

    box4 = np.array([[20, 20, 30, 30]])
    assert np.isclose(box_ious(box1, box4)[0, 0], 0.0)


# 3. Kalman Filter: initiation, prediction, update
def test_kalman_filter():
    kf = KalmanFilter()
    measurement = np.array([100.0, 200.0, 40.0, 80.0]) # cx, cy, w, h
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert np.allclose(mean[:4], measurement)

    # Predict
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)

    # Update with new measurement moving right
    new_meas = np.array([105.0, 200.0, 40.0, 80.0])
    upd_mean, upd_cov = kf.update(pred_mean, pred_cov, new_meas)
    assert upd_mean[0] > 100.0


# 4. Hungarian assignment
def test_hungarian_assignment():
    # Cost matrix: 2 tracks, 2 detections
    cost = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    matches, unassigned_rows, unassigned_cols = linear_assignment(cost, threshold=0.5)
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert len(unassigned_rows) == 0
    assert len(unassigned_cols) == 0

    # Test threshold rejection
    cost_high = np.array([[0.9]])
    matches, unassigned_rows, unassigned_cols = linear_assignment(cost_high, threshold=0.5)
    assert len(matches) == 0
    assert 0 in unassigned_rows
    assert 0 in unassigned_cols


# 5. Feature Gallery and Cosine Distance
def test_feature_gallery():
    gallery = FeatureGallery(max_samples=4, ema_momentum=0.90)
    assert gallery.is_empty()

    f1 = np.ones(512, dtype=np.float32)
    gallery.add_initial_feature(f1)
    assert not gallery.is_empty()

    # Identical feature distance must be ~0.0
    assert np.isclose(gallery.distance_to(f1), 0.0, atol=1e-5)

    # Orthogonal feature
    f_orth = np.zeros(512, dtype=np.float32)
    f_orth[0] = 1.0
    f1_zero = np.zeros(512, dtype=np.float32)
    f1_zero[1] = 1.0
    gallery_orth = FeatureGallery()
    gallery_orth.add_initial_feature(f_orth)
    assert np.isclose(gallery_orth.distance_to(f1_zero), 1.0, atol=1e-5)


# 6. Track lifecycle: NEW -> TRACKED -> LOST -> EXPIRED
def test_track_lifecycle():
    Track.reset_counter()
    manager = TrackManager(track_buffer=5, confirm_window=2)

    # Frame 1: New detection creates candidate
    t1 = Track(box=np.array([10, 10, 50, 50]), score=0.9)
    manager.add_new_candidate(t1)
    manager.step_lifecycle()

    # After 1 detection, t1 should still be in new_tracks (not yet confirmed)
    assert len(manager.get_output_tracks()) == 0

    # Frame 2: Matched again -> promoted to TRACKED
    t1.update(box=np.array([12, 12, 52, 52]), score=0.9, frame_id=2)
    manager.update_confirmed(t1)
    manager.step_lifecycle()

    # Now confirmed
    assert t1 in manager.tracked_tracks
    assert len(manager.get_output_tracks()) == 1

    # Frames 3-8: Missing -> moves to LOST, then EXPIRED after buffer
    for frame in range(3, 10):
        manager.predict_all()
        manager.step_lifecycle()

    assert t1.is_expired
    assert len(manager.all_active_tracks) == 0


# 7. Directional Line Counter: IN and OUT crossing
def test_line_counter():
    # Vertical line at x=100 from y=0 to y=200, normal points right (+x is IN)
    counter = DirectionalLineCounter(line_start=(100, 0), line_end=(100, 200), cooldown_frames=5)

    Track.reset_counter()
    t = Track(box=np.array([70, 40, 90, 80]), score=0.9)
    t.state = TrackState.TRACKED # Manually confirm

    # Step 1: Start at x=80
    counter.update([t], frame_id=1)
    assert counter.in_count == 0

    # Step 2: Cross line to x=120 (moved left to right -> IN)
    t.update(box=np.array([110, 40, 130, 80]), score=0.9, frame_id=2)
    counter.update([t], frame_id=2)
    assert counter.in_count == 1
    assert counter.out_count == 0

    # Step 3: Re-cross back to left (moved right to left -> OUT after cooldown)
    t.update(box=np.array([70, 40, 90, 80]), score=0.9, frame_id=10)
    counter.update([t], frame_id=10)
    assert counter.in_count == 1
    assert counter.out_count == 1


# 8. Empty frames and edge cases
def test_tracker_empty_frames():
    tracker = ByteTracker()
    tracks = tracker.update([], frame=None)
    assert len(tracks) == 0

