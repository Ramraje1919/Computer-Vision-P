# MOT17 Baseline Failure Case Studies and Root Cause Analysis

This document details five representative failure cases observed during the audit of the baseline tracking system on `MOT17-02-FRCNN`.

---

### Case 1: ID Switch During Pedestrian Crossing
- **Sequence**: `MOT17-02-FRCNN`
- **Frame Range**: Frames 142 – 168
- **Tracks Involved**: Target A (Track ID 4, walking left-to-right) and Target B (Track ID 7, walking right-to-left)
- **Failure Type**: Identity Switch (ID Swap)
- **Observed Behavior**: At frame 150, Target A and Target B cross paths. Bounding boxes overlap with $\text{IoU} \approx 0.45$. In frame 154, the Hungarian matcher associates Track 4 to Target B and Track 7 to Target A.
- **Root Cause**:
  1. The baseline uses pure spatial IoU and center distance without appearance validation.
  2. The Kalman filter predicts a linear trajectory through the intersection, but spatial overlap is identical for both hypotheses.
  3. No trajectory momentum or directional gating is applied.
- **Proposed Correction**:
  - Integrate OSNet appearance cosine distance into the association cost: $C_{fuse} = \alpha C_{app} + (1 - \alpha) C_{IoU}$.
  - Apply motion momentum consistency (OCM) to penalize abrupt $180^\circ$ velocity inversions.

---

### Case 2: Failure to Retain ID Across Temporary Occlusion (Behind Pillar)
- **Sequence**: `MOT17-02-FRCNN`
- **Frame Range**: Frames 210 – 245
- **Tracks Involved**: Ground Truth ID 12
- **Failure Type**: Track Fragmentation / Lost Identity Re-assignment
- **Observed Behavior**: At frame 215, GT 12 walks behind a large architectural pillar and disappears. Track 12 is placed in missing state. At frame 230 (15 frames later), `max_missed = 15` expires and Track 12 is deleted. At frame 238, the person steps out from behind the pillar. The detector generates a detection, and the tracker assigns a brand new ID: Track 48.
- **Root Cause**:
  1. Extremely short track retention buffer (`max_missed = 15`, only 0.5s).
  2. Complete deletion of track memory upon buffer expiration.
  3. No re-entry candidate gallery.
- **Proposed Correction**:
  - Extend the lost track buffer to 90–150 frames.
  - Implement a dedicated `LOST` track pool.
  - Retain historical appearance signatures in a multi-sample gallery to re-identify the person upon emergence.

---

### Case 3: Track Proliferation From Single-Frame False Detections
- **Sequence**: `MOT17-02-FRCNN`
- **Frame Range**: Frame 45 – 52
- **Tracks Involved**: Ephemeral Tracks 18, 19, 21
- **Failure Type**: False Positive Track Initiation
- **Observed Behavior**: The YOLO detector fires spurious person detections on a mall advertising banner and luggage cart for 1 frame. The baseline tracker immediately initializes Tracks 18 and 19. On the subsequent frame, the detection disappears, and the tracks linger for 15 frames as misses.
- **Root Cause**:
  - Unconfirmed detections are immediately elevated to full track status with new global IDs.
- **Proposed Correction**:
  - Require a two-frame or three-frame confirmation hysteresis: new candidates start in `NEW` state and must match consecutive detections before promotion to `TRACKED` and ID emission.

---

### Case 4: Missed Distant Pedestrians Due to Hard Confidence Cutoff
- **Sequence**: `MOT17-02-FRCNN`
- **Frame Range**: Continuous throughout sequence (frames 1 – 600)
- **Tracks Involved**: Distant pedestrians in background concourse (GT IDs 15, 18, 22, 29)
- **Failure Type**: Severe False Negatives (FN = 11,815)
- **Observed Behavior**: Background pedestrians appear small ($h < 50$ px) with low contrast, producing YOLO confidence scores between $0.08$ and $0.18$. The baseline cutoff at $0.20$ completely discards these detections, resulting in an overall detection recall of only 35.2%.
- **Root Cause**:
  - Single-stage confidence threshold discarding low-confidence bounding boxes.
- **Proposed Correction**:
  - Implement two-stage BYTE association: divide detections into $D_{high} (\ge 0.50)$ and $D_{low} (0.10 \le \text{conf} < 0.50)$.
  - Use $D_{low}$ boxes to recover tracks already initiated, reclaiming occluded and distant pedestrians without increasing false positive tracks.

---

### Case 5: Appearance Gallery Corruption During Partial Occlusion
- **Sequence**: `MOT17-02-FRCNN`
- **Frame Range**: Frames 320 – 345
- **Tracks Involved**: Track 14
- **Failure Type**: Re-ID Failure Due to Contaminated Gallery
- **Observed Behavior**: Track 14 is partially occluded by a shopping cart for 10 frames. In `tracker_reid.py`, the EMA gallery updates every frame: $e = 0.90 \cdot e + 0.10 \cdot f_{crop}$. The crop contains 60% shopping cart pixels. When the target clears the occlusion at frame 342, the gallery embedding is heavily shifted toward the cart. The cosine similarity to the clean pedestrian drops to 0.42 (below threshold), and the tracker fails to associate, spawning Track 53.
- **Root Cause**:
  - Blind single-vector EMA updating without occlusion quality gating or multiple exemplar storage.
- **Proposed Correction**:
  - Implement an occlusion-aware feature gallery:
    1. Only update the gallery when detection confidence is high and visibility/aspect ratio indicates an unoccluded full-body crop.
    2. Maintain a multi-sample exemplar memory (e.g., initial clean exemplar + running representative cluster) rather than a single corrupted running average.

