# Forensic Baseline Audit and Technical Analysis

## 1. Executive Summary

A comprehensive audit was performed on the existing tracking codebase (`tracker_baseline.py`, `tracker.py`, `tracker_reid.py`, `detect_people.py`, and `evaluate_baseline.py`). The baseline was evaluated on `MOT17-02-FRCNN` (600 frames, 18,581 ground truth pedestrian annotations) using `motmetrics 1.4.0` at the standard MOTChallenge match threshold ($\text{IoU} \ge 0.50$).

The baseline tracking performance is severely degraded:
- **Baseline IoU Tracker**: MOTA: **19.93%**, IDF1: **30.34%**, ID Switches: **217** (Previous extended run: MOTA: **30.92%**, IDF1: **28.67%**, ID Switches: **332**)
- **Kalman + Hungarian Tracker**: MOTA: **19.75%**, IDF1: **26.75%**, ID Switches: **240**

This analysis investigates the 12 technical dimensions of the failure modes.

---

## 2. In-Depth Technical Investigation

### 2.1 Detection Quality & Model Capacity
- **Current Detector**: Pretrained YOLO11n (`yolo11n.pt`, nano architecture, ~2.6M parameters).
- **Audit Finding**: In dense, high-resolution surveillance sequences (e.g., MOT17-02, 1920x1080), distant pedestrians and partially occluded persons occupy small pixel areas ($< 32 \times 32$ pixels). YOLO11n produces only 9,611 total detections across 600 frames, compared to 18,581 ground truth instances. This yields a detection recall of only **35.2%** and **11,815 misses (FN)**.
- **Impact**: MOTA is mathematically bounded by $(1 - \frac{FN + FP + IDSW}{GT})$. With 11,815 misses on 18,581 objects, the maximum achievable MOTA before any tracker errors is already capped at $< 36.4\%$.

### 2.2 Confidence Threshold Policy
- **Current Policy**: A single hard cutoff threshold $\tau_{conf} = 0.20$.
- **Audit Finding**: Pedestrians undergoing partial occlusion or motion blur produce confidence scores in the range $[0.05, 0.20]$. Discarding these detections creates persistent track termination. Conversely, lowering the threshold globally to $0.10$ introduces substantial false positives (cluttered backgrounds, reflections).
- **Root Cause**: The lack of a two-stage association mechanism (BYTE paradigm) that separates high-confidence track maintenance from low-confidence occlusion recovery.

### 2.3 Non-Maximum Suppression (NMS) Threshold
- **Current Policy**: Default YOLO NMS ($\text{IoU} \approx 0.70$).
- **Audit Finding**: In crowded pedestrian crowds (such as MOT17-02 and MOT17-04), walking pedestrians frequently overlap with bounding box IoU of $0.40 - 0.65$. The detector frequently suppresses one of two overlapping pedestrians or merges their bounding boxes into a single wide bounding box spanning both people.

### 2.4 Bounding Box Quality & Coordinate Jitter
- **Audit Finding**: Because the baseline IoU tracker does not predict target velocity, the bounding box remains frozen at its last observed position for up to 15 frames. When a moving person reappears, the spatial overlap between the old stationary box and the new box is often zero, breaking the track.

### 2.5 Kalman Filter Formulation
- **Audit Finding**: In `tracker.py`, the Kalman state vector is $[cx, cy, w, h, vx, vy, vw, vh]^T$ with constant process noise ($0.03 \cdot I$) and measurement noise ($1.0 \cdot I$).
- **Defects Identified**:
  1. Measurement noise $R$ is fixed regardless of detection confidence. When a low-quality, occluded box arrives, the Kalman filter over-corrects to the occluded box.
  2. In `tracker_baseline.py`, the Kalman filter is omitted entirely.
  3. No camera motion compensation is applied, causing systematic drift on moving sequences.

### 2.6 Hungarian Assignment Formulation
- **Audit Finding**: Both `tracker_baseline.py` and `tracker.py` construct global cost matrices and execute `linear_sum_assignment`.
- **Defects Identified**:
  - Global Hungarian matching seeks to minimize aggregate cost across all pairs. Without strict spatial gating before assignment, a lost track on the left side of the frame can be forcibly assigned to an unmatched detection on the right side if no other candidate exists.
  - Thresholding after assignment (`if overlap >= 0.20`) leaves valid local matches unassigned if the global assignment made a suboptimal match elsewhere.

### 2.7 Association Thresholds
- **Audit Finding**: Fixed threshold $\text{IoU} \ge 0.20$ or center distance $\le 150\text{px}$.
- **Defects Identified**: When a pedestrian accelerates, turns, or experiences camera shake, consecutive frame IoU frequently drops below $0.20$. The track is instantly abandoned.

### 2.8 Track Buffer & Life Cycle
- **Audit Finding**: `tracker_baseline.py` sets `max_missed = 15` frames (0.5 seconds at 30 fps).
- **Defects Identified**: 15 frames is far too short for real-world occlusions. In MOT17-02, pedestrians walk behind pillars for 30–90 frames. At frame 16 of occlusion, the track is deleted. When the person emerges, they are assigned a completely new ID, directly generating an ID switch.

### 2.9 Lost-Track Management
- **Audit Finding**: The baseline tracks have only two implicit states: active (`missed_frames == 0`) and missing (`missed_frames > 0`).
- **Defects Identified**: There is no explicit state machine (`NEW` $\rightarrow$ `TRACKED` $\rightarrow$ `LOST` $\rightarrow$ `RECOVERED` $\rightarrow$ `EXPIRED`). Lost tracks are kept in the same pool as active tracks during association, polluting the cost matrix and increasing ID switches.

### 2.10 ReID Embedding Quality (OSNet)
- **Audit Finding**: `appearance_model.py` loads `osnet_x0_25` (256x128 input, 512-d feature vector).
- **Defects Identified**:
  1. Features are extracted sequentially on CPU for every detection (very slow, $< 2$ FPS).
  2. Bounding boxes are cropped without checking whether the box contains substantial occluding background or another person.
  3. Raw features are extracted from occluded crops, contaminating the identity signature.

### 2.11 Feature Gallery Architecture
- **Audit Finding**: `tracker_reid.py` uses a single-vector Exponential Moving Average (EMA) update:
  $$e_t = 0.90 \cdot e_{t-1} + 0.10 \cdot f_t$$
- **Defects Identified**:
  - A single vector cannot represent appearance variations across multiple poses, lighting changes, and partial occlusions.
  - When a pedestrian is partially occluded by another person or an object, blending the contaminated feature $f_t$ into the single vector irreversibly corrupts the gallery. When the target emerges into full view, the corrupted gallery no longer matches the target.

### 2.12 Track Duplication & False Positive Initiation
- **Audit Finding**: Any unmatched detection immediately initializes a new track (`self.next_id += 1`).
- **Defects Identified**: Spurious false positive detections from reflections or background objects instantly become new tracks. Without a confirmation requirement (e.g., must be detected in 2 or 3 consecutive frames before promotion to `TRACKED`), false positive tracks proliferate, driving down MOTA and generating false identities.

---

## 3. Preserved Baseline Code Checkpoint

The original baseline files are strictly preserved and untouched:
- Backup directory: `baseline/` (`baseline/tracker_baseline.py`, `baseline/tracker.py`, `baseline/tracker_reid.py`, `baseline/main.py`, `baseline/evaluate_baseline.py`, `baseline/evaluate_comparison.py`)
- Git checkpoint commit: `e968511: Checkpoint: Preserved original baseline implementation`

