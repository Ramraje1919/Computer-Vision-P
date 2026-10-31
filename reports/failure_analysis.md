# Failure Analysis — MOT17 Multi-Object Tracking Pipeline

> Tracker: ByteTrack (tuned) · Detector: MOT17 Public Detections (SDP variant)
> Evaluation sequences: validation split (02, 05, 09) + held-out split (04, 10, 11, 13)

---

## 1. High-Level Failure Taxonomy

| Failure Mode | Primary Cause | Metric Impact | Severity |
|---|---|---|---|
| Long occlusion → ID loss | `track_buffer` exhausted | +IDSW, +FN | High |
| Crossing trajectories → ID swap | IoU ambiguity in dense frame | +IDSW | High |
| Missed detections (FN) | Detector recall ceiling | +FN, -MOTA | Very High |
| False positives from detector | Low-conf detections passing stage 2 | +FP, -MOTA | Medium |
| Fragmentation on re-entry | Track killed, re-born with new ID | +IDSW, +frag | Medium |
| Scale/depth plane mismatch | Near/far pedestrians, IoU degrades | +IDSW | Medium |
| Moving-camera drift | No CMC; bounding-box displacement | +FN, +IDSW | Medium |

---

## 2. Detailed Failure Mode Analysis

### 2.1 Detection Recall Ceiling (Dominant Failure)

The single largest source of errors across all sequences is **missed detections (FN)**.
Public MOT17 SDP detections have high precision but limited recall in crowded or heavily
occluded regions.

**Evidence:**
- MOT17-04-SDP (1050 frames, extreme density): FN = 11,762 — the tracker receives no input
  signal for many ground-truth objects and cannot track what it never detects.
- MOT17-09-SDP (150-frame demo): FN = 4,522 / GT = 5,325 → only 15% of objects detected
  in the first 150 frames when the sequence is still "warming up".

**Why the tracker can't fix it:** ByteTrack's second-stage low-confidence association recovers
some detections missed in stage 1, but only if a detection exists at all. If the detector
produces zero box for a pedestrian, no tracking algorithm can recover that pedestrian.

**Mitigation (future work):** Use a stronger detector (YOLOv8-X, RT-DETR, or the MOT17
DPM/FRCNN ensemble) or train a sequence-specific detector head.

---

### 2.2 Long-Occlusion Track Loss (ID Switches)

When two or more pedestrians occlude each other for longer than `track_buffer=90` frames
(≈ 3 seconds at 30 fps), the occluded track expires. When the pedestrian re-emerges,
a new track ID is assigned → **IDSW**.

**Evidence:**
- MOT17-10-SDP (moving camera, night): IDSW = 304 — the highest in the held-out set.
  Moving-camera sequences have more sudden detection gaps (frame blur, camera pan) that push
  tracks over the buffer limit.
- MOT17-11-SDP (indoor mall, moving): IDSW = 85, frag = 54 on the full sequence evaluation.

**Key insight:** `track_buffer=90` was the optimal value on the validation split (static
cameras). On moving-camera sequences the same buffer is insufficient because detection
gaps are more frequent and longer.

**Mitigation:** Sequence-adaptive `track_buffer` (longer for moving-camera), or GMC
(Global Motion Compensation / Camera Motion Compensation) as used in BoT-SORT to stabilise
predicted positions during camera pans.

---

### 2.3 Crossing Trajectories — IoU Ambiguity

When two pedestrians walk toward each other and cross paths, their bounding boxes
temporarily overlap. The Hungarian solver minimises total cost, but with near-identical
IoU distances to both IDs, small perturbations (detection jitter, Kalman prediction error)
cause an ID swap.

**Evidence:**
- This is the primary cause of IDSW on MOT17-02-SDP (dense indoor, baseline IDSW = 217;
  tuned ByteTrack IDSW = 205) and MOT17-13-SDP (IDSW = 177).
- The ablation study showed that OSNet ReID (intended to resolve this ambiguity via
  appearance) made it *worse* on public detections because the fused cost matrix
  amplified appearance errors in homogeneous-looking pedestrian crowds.

**Mitigation:** Velocity-gated ReID — use appearance only when IoU drops below 0.3
(i.e., after crossing), and otherwise trust motion. BoT-SORT's CMC also partially
resolves this by aligning predictions with camera motion.

---

### 2.4 Moving-Camera Drift (MOT17-10, 11)

ByteTrack uses a Kalman filter with **constant-velocity kinematics in pixel space**.
When the camera pans or tilts, all pedestrians shift in the frame simultaneously, but
the Kalman predictions remain anchored to the previous frame's pixel positions.
This causes IoU between prediction and detection to drop sharply → tracks miss their
correct detection → association failure.

**Evidence:**
- MOT17-10-SDP: MOTA = 66.05%, IDSW = 304. Despite having fewer GT objects than MOT17-04,
  it has the highest IDSW count, consistent with camera-motion-induced drift.
- MOT17-11-SDP: MOTA = 70.18%, IDSW = 85 (better because the camera motion is slower).

**Mitigation:** Global Motion Compensation (GMC / CMC) — estimate the homography between
consecutive frames and warp Kalman predictions before association. BoT-SORT includes this
natively (disabled by default in our experiments because it is expensive on CPU).

---

### 2.5 False Positives from Low-Confidence Stage 2

ByteTrack's second-stage association uses detections in the range `[conf_low=0.10, conf_high=0.50)`.
This recovers partially-occluded pedestrians, but also lets through false positives
(reflections, lamp posts, vehicle parts).

**Evidence:**
- MOT17-04-SDP: FP = 105 (low given 1050 frames, detector is precise).
- MOT17-13-SDP: FP = 486 — the bus viewpoint and street clutter generate more false
  detector boxes than static indoor scenes.

**Mitigation:** Scene-specific confidence tuning or a post-hoc validity filter
(aspect ratio, minimum box area, proximity to scene boundary).

---

### 2.6 ReID Appearance Degradation (Ablation Finding)

When OSNet ReID was integrated into the hybrid tracker, IDF1 **dropped** from 45.93% to
29.14% on MOT17-02-SDP (Ablation Study, Stage G). Root causes:

1. **Pedestrian appearance homogeneity**: People in the same sequence often wear similar
   dark winter clothing. OSNet's 128-D embedding space cannot discriminate them reliably
   at `osnet_x0_25` capacity.
2. **Fused cost matrix dominance**: With 60% appearance weight, IoU-guided motion cues
   (which are correct in dense scenes) are overridden by noisy appearance distances.
3. **CPU latency**: OSNet inference at ~3 FPS on CPU-only hardware forces frame
   downsampling, degrading feature quality.

**Mitigation:** Use ReID only as a fallback tiebreaker (weight ≤ 0.2), gate it on
low-IoU frames only, or upgrade to GPU inference for reliable appearance features.

---

## 3. Sequence-Level Failure Profile

| Sequence | Primary Failure | Secondary Failure |
|---|---|---|
| MOT17-02-SDP (validation) | Crossing trajectories | Long occlusion |
| MOT17-05-SDP (validation) | Moving camera drift | Low-density but tiny boxes |
| MOT17-09-SDP (validation) | Detection recall (sparse scene, GT-detector mismatch) | None significant |
| MOT17-04-SDP (held-out) | Detection recall (extreme density) | Occlusion |
| MOT17-10-SDP (held-out) | Moving camera drift | Long occlusion |
| MOT17-11-SDP (held-out) | Moving camera drift | Fragmentation |
| MOT17-13-SDP (held-out) | False positives (clutter) | Crossing trajectories |

---

## 4. Quantitative Error Budget

Based on the held-out evaluation (4 sequences, 3,354 total frames):

| Error Type | Count | % of Total Errors |
|---|---|---|
| False Negatives (FN) | 22,947 | ~93% |
| False Positives (FP) | 1,311 | ~5% |
| ID Switches (IDSW) | 654 | ~2% |

**Takeaway:** 93% of all errors come from missed detections — not from the tracker.
Improving the detector would have a dramatically larger impact on MOTA than any
tracker algorithm improvement.

---

## 5. Recommendations for Future Work

1. **Stronger detector**: Replace public MOT17 detections with YOLO11-L or RT-DETR finetuned
   on MOT17-train for higher recall without sacrificing precision.
2. **Camera Motion Compensation**: Enable GMC in BoT-SORT for sequences 10, 11 (confirmed
   moving-camera); this is the single highest-impact algorithmic change for IDSW on those seqs.
3. **Adaptive track buffer**: Use sequence metadata (seqinfo.ini `imHz`, camera type) to
   set `track_buffer` dynamically — longer buffers for moving-camera sequences.
4. **Velocity-gated ReID**: Apply appearance features only when predicted IoU < 0.3 to
   act as a tiebreaker after motion cues alone are ambiguous.
5. **Ensemble association**: Combine detections from multiple detector variants (DPM + FRCNN + SDP)
   via NMS to boost recall at a given precision threshold.

---

*Generated for Stage J of the MOT17 Multi-Object Tracking Project.*

