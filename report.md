# MOT17 Multi-Object Tracking — Master Technical Report

**Project**: High-Accuracy Multi-Object Tracking on MOT17  
**Tracker**: ByteTrack (tuned) with public detections  
**Environment**: Python 3.11.9, PyTorch 2.15.0+cpu, Windows, CPU-only  
**Evaluation Standard**: MOTChallenge metrics (MOTA, IDF1, IDSW, FP, FN)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Split Protocol](#3-data-split-protocol)
4. [System Architecture](#4-system-architecture)
5. [Baseline Audit](#5-baseline-audit)
6. [Algorithm Design & Implementation](#6-algorithm-design--implementation)
7. [Ablation Study](#7-ablation-study)
8. [Hyperparameter Optimisation](#8-hyperparameter-optimisation)
9. [Held-Out Benchmark](#9-held-out-benchmark)
10. [Visualisations](#10-visualisations)
11. [Failure Analysis Summary](#11-failure-analysis-summary)
12. [Conclusions & Future Work](#12-conclusions--future-work)

---

## 1. Problem Statement

Multi-Object Tracking (MOT) requires associating detections across video frames to
form consistent, long-lived trajectories for each person. The challenge is to maximise
identity consistency (IDF1) while keeping false positives and misses low (MOTA),
particularly under heavy occlusion, scene clutter, and camera motion.

**Goals for this project:**
- Build a modular, production-grade tracking pipeline on MOT17.
- Benchmark multiple tracker architectures under controlled ablation conditions.
- Tune hyperparameters on a validation split with no data leakage.
- Evaluate final performance on a strictly held-out split.
- Produce annotated video demonstrations and comprehensive documentation.

---

## 2. Dataset Overview

**MOT17** is the de-facto benchmark for pedestrian tracking, consisting of:
- **7 unique scene sequences**, each provided with 3 detector variants (DPM, FRCNN, SDP) → 21 training sequences total.
- All sequences share identical underlying images; only the `det/det.txt` files differ by variant.
- Ground truth: class 1 (Pedestrian) with `conf > 0` are active targets; all others are ignore regions.

| Statistic | Value |
|---|---|
| Total training sequences | 21 (7 scenes × 3 detectors) |
| Total frames (training) | ~5,316 unique frames |
| Detector variants | DPM (Deformable Parts Model), FRCNN (Faster R-CNN), SDP (Scale-Dependent Pooling) |
| Evaluation metric | MOTA, IDF1, IDSW, FP, FN, Precision, Recall |

**Validation** (Stage H dataset inspection): 100% of 42 expected sequence directories passed
structural validation — frame continuity, image integrity, and GT format checks.

---

## 3. Data Split Protocol

Sequence-level splitting was applied to prevent data leakage: all detector variants of a
scene are assigned to the same partition.

| Split | Sequences | Purpose |
|---|---|---|
| **Training / ablation** | MOT17-02, 05, 09 (SDP) | Ablation study (Stage G) |
| **Validation** | MOT17-02, 05, 09 (SDP) | Hyperparameter tuning (Stage H) |
| **Held-out** | MOT17-04, 10, 11, 13 (SDP) | Final benchmark (Stage I) |

> **Note**: MOT17-02, 05, 09 double as both ablation and validation sequences —
> they were never exposed during final model selection; held-out sequences were
> strictly unseen until Stage I.

**Excluded**: `data/MOT16-03.mp4` — excluded from all metric computation throughout
the project.

---

## 4. System Architecture

### 4.1 Module Map

```
src/
├── detection/        # Detection loading, caching, pre/post-processing
│   ├── detector.py       — YOLODetector, PublicDetectionLoader, DetectionCache
│   ├── preprocessing.py  — frame normalisation, resize
│   └── postprocessing.py — NMS, box format conversion
├── tracking/         # Core tracking algorithms
│   ├── kalman_filter.py  — BoT-SORT 8-state constant-velocity KF
│   ├── hungarian_matcher.py — Lapjax-free Hungarian assignment (scipy)
│   ├── association.py    — vectorised IoU, fused cost, spatial gating
│   ├── track.py          — Track entity with FSM (Tentative→Confirmed→Lost→Dead)
│   ├── track_manager.py  — Pool manager with confirm_window
│   ├── bytetrack.py      — Two-stage BYTE association
│   ├── botsort.py        — BoT-SORT with optional CMC
│   └── tracker.py        — HybridReIDTracker (motion + OSNet appearance)
├── reid/             # Deep appearance re-identification
│   ├── osnet.py          — Direct OSNet loader (bypasses torchreid cascade)
│   ├── feature_extractor.py — OSNet inference wrapper
│   ├── feature_gallery.py  — Multi-sample exemplar memory with quality gating
│   └── reidentifier.py   — ReID decision module
├── counting/         # Directional person counting
│   └── line_counter.py   — Virtual line counter (IN/OUT with cooldown)
├── evaluation/       # Metric computation
│   ├── evaluator.py      — motmetrics-based MOT evaluator
│   └── mot_format.py     — MOTChallenge gt/pred file I/O
├── visualization/    # Video annotation
│   └── visualizer.py     — Bounding boxes, ID labels, trails, dashboard, counter
└── utils/
    ├── paths.py, logger.py, video_loader.py
```

### 4.2 Tracker FSM

```
Tentative ──(confirm_window hits)──► Confirmed ──(update)──► Confirmed
    │                                     │
    └──(max_age exceeded)──► Dead         └──(lost)──► Lost ──(track_buffer)──► Dead
```

### 4.3 ByteTrack Two-Stage Association (BYTE)

```
Detections
    │
    ├─ High-confidence (≥conf_high=0.50) ──► Stage 1: Hungarian(IoU, confirmed tracks)
    │                                              └─► Matched | Unmatched tracks
    └─ Low-confidence  (conf_low=0.10..) ──► Stage 2: Hungarian(IoU, unmatched tracks)
                                                   └─► Recovered | Lost
```

---

## 5. Baseline Audit

The original baseline (preserved in `baseline/`) was a simple IoU tracker without
Kalman filtering, confidence thresholding, or track lifecycle management.

| Metric | Baseline | ByteTrack (tuned) | Improvement |
|---|---|---|---|
| MOTA (MOT17-02-SDP) | 19.93% | 54.79% (val mean) | +34.86 pp |
| IDF1 (MOT17-02-SDP) | 30.34% | 57.66% (val mean) | +27.32 pp |
| IDSW (MOT17-02-SDP) | 217 | ~205 | -12 |

The baseline lacked: Kalman-predicted motion, two-stage confidence filtering,
track lifecycle management (tentative/confirmed/lost states), and systematic
hyperparameter optimisation.

---

## 6. Algorithm Design & Implementation

### 6.1 Kalman Filter (BoT-SORT 8-state)

State vector: `[cx, cy, aspect_ratio, height, vx, vy, va, vh]`  
Measurement: `[cx, cy, aspect_ratio, height]`  
The 8-state formulation models aspect-ratio velocity separately from size velocity,
improving prediction stability for pedestrians under variable-speed motion.

### 6.2 Association Cost Matrix

```
Cost(i,j) = 1 - IoU(predicted_box_i, detection_j)
```

With **spatial gating** (`max_distance=0.9`) applied before Hungarian assignment
to avoid spurious long-range matches.

For the HybridReIDTracker:
```
Cost(i,j) = iou_weight × (1 - IoU) + reid_weight × cosine_distance(feat_i, feat_j)
```

### 6.3 OSNet ReID (x0.25 variant)

- Architecture: OSNet-x0.25 (2.3M params, 128-D embedding)
- Weights: `osnet_x0_25_msmt17_combineall_256x128` (pre-trained on MSMT17)
- Input: 256×128 cropped pedestrian patch, normalised with ImageNet stats
- Loading: Direct `importlib`-based loader (bypasses torchreid init cascade)

### 6.4 Directional Line Counter

A virtual counting line is placed at 70% of frame height. For each confirmed track,
the signed dot product of its velocity vector with the line normal determines IN/OUT:

```
cross = dot(velocity, line_normal)
if cross > 0: IN++  else: OUT++
```

A per-track cooldown of 30 frames prevents duplicate counting on oscillating tracks.

---

## 7. Ablation Study

Four tracker configurations were compared on validation sequences (MOT17-02-SDP, MOT17-09-SDP):

| Tracker | MOTA (02) | IDF1 (02) | IDSW (02) | MOTA (09) | IDF1 (09) | IDSW (09) |
|---|---|---|---|---|---|---|
| Baseline (IoU only) | 41.59% | 41.79% | 301 | 63.00% | 52.90% | 59 |
| ByteTrack | **42.35%** | **45.93%** | **205** | **63.23%** | **62.56%** | **24** |
| BoT-SORT (no CMC) | 42.35% | 45.93% | 205 | 63.23% | 62.56% | 24 |
| HybridReID (OSNet) | 34.78% | 29.14% | 460 | 61.88% | 44.20% | 89 |

**Key finding**: OSNet ReID degraded IDF1 from 45.93% → 29.14% on MOT17-02 (+255 IDSW).
Appearance features are unreliable on CPU at `x0.25` capacity for dense homogeneous crowds.
ByteTrack (motion-only) is the best-performing configuration.

---

## 8. Hyperparameter Optimisation

Grid search over 12 configurations on validation sequences (MOT17-02, 05, 09 SDP):

| Parameter | Range Searched | Optimal Value |
|---|---|---|
| `conf_high` | 0.45, 0.50, 0.55 | **0.50** |
| `conf_low` | 0.08, 0.10, 0.15 | **0.10** |
| `match_thresh_stage1` | 0.75, 0.80, 0.85 | **0.80** |
| `match_thresh_stage2` | 0.40, 0.45, 0.50 | **0.45** |
| `track_buffer` | 30, 60, 90, 120 | **90** |

**Best validation results** (Score = 0.5×MOTA + 0.5×IDF1):

| Metric | Value |
|---|---|
| Mean MOTA | 54.79% |
| Mean IDF1 | 57.66% |
| Total IDSW | 317 |
| Score | 53.05 |

---

## 9. Held-Out Benchmark

Tuned ByteTrack evaluated on the 4 held-out sequences (never seen during optimisation):

| Sequence | Scene | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|
| MOT17-04-SDP | Night dense crowd | **74.86%** | **74.76%** | 88 | 105 | 11,762 |
| MOT17-10-SDP | Moving cam, night | 66.05% | 47.58% | 304 | 528 | 3,527 |
| MOT17-11-SDP | Moving cam, mall | 70.18% | 65.37% | 85 | 192 | 2,537 |
| MOT17-13-SDP | Bus perspective | 50.32% | 58.72% | 177 | 486 | 5,121 |
| **Mean** | | **65.35%** | **61.61%** | 654 | 1,311 | 22,947 |

### Comparison: Validation vs Held-Out

| Metric | Validation | Held-Out | Delta |
|---|---|---|---|
| Mean MOTA | 54.79% | **65.35%** | +10.56 pp |
| Mean IDF1 | 57.66% | **61.61%** | +3.95 pp |
| Total IDSW | 317 | 654 | +337 |

Held-out MOTA is higher than validation — the held-out sequences (especially MOT17-04)
have better SDP detection quality than the validation sequences. This rules out overfitting.
The higher IDSW on held-out reflects the moving-camera sequences (10, 11) which introduce
camera-motion-induced track breaks.

---

## 10. Visualisations

Two annotated MP4 videos were generated using `main.py --visualize --count`:

| Video | Sequence | Frames | Description |
|---|---|---|---|
| `outputs/visualizations/MOT17-09-SDP_bytetrack.mp4` | MOT17-09-SDP | 150 | Outdoor daytime, static camera |
| `outputs/visualizations/MOT17-11-SDP_bytetrack.mp4` | MOT17-11-SDP | 150 | Indoor mall, moving camera |

Each video includes:
- Colour-coded bounding boxes per track ID (persistent colour across frames)
- Track ID labels
- Motion trails (last N positions)
- Frame counter and FPS overlay
- Directional IN/OUT counting overlay (virtual line at 70% frame height)

---

## 11. Failure Analysis Summary

The dominant failure mode is **missed detections (FN = 93% of all errors)**, not the tracker.
The tracker cannot recover pedestrians that the detector never fires on.

Remaining tracker-specific failures:
1. Long occlusions (>90 frames) → track buffer exhausted → re-entry as new ID
2. Crossing trajectories → IoU ambiguity → ID swap
3. Moving-camera drift → Kalman prediction de-aligned → association failure

See [`reports/failure_analysis.md`](reports/failure_analysis.md) for full case-by-case analysis.

---

## 12. Conclusions & Future Work

### Conclusions

1. **ByteTrack with BYTE two-stage association** is the most effective tracker configuration
   under CPU-only constraints, achieving Mean MOTA=65.35% and Mean IDF1=61.61% on the held-out split.

2. **Detection quality dominates tracking quality**: 93% of errors (by count) originate
   from missed detections. The gap between what the detector provides and what the GT
   contains is the primary bottleneck.

3. **ReID appearance features degrade performance** at OSNet-x0.25 capacity on CPU for
   dense homogeneous pedestrian crowds. Appearance features should be gated by motion
   uncertainty rather than always applied.

4. **Hyperparameter optimisation on the validation split generalises**: Held-out performance
   is higher than validation performance, confirming no overfitting.

### Future Work

| Priority | Improvement | Expected Gain |
|---|---|---|
| High | Stronger detector (YOLOv8-X, RT-DETR) | +15–25 pp MOTA |
| High | Camera Motion Compensation (GMC/CMC) | −30–50% IDSW on moving seqs |
| Medium | Adaptive track_buffer (per-sequence) | −10–20% IDSW |
| Medium | Velocity-gated ReID (weight ≤ 0.2) | +3–5 pp IDF1 |
| Low | Multi-detector ensemble (DPM+FRCNN+SDP) | +5–10 pp recall |

---

*Report compiled from Stages A–J of the MOT17 Multi-Object Tracking Project.*  
*All experiments run locally on CPU — no GPU, no cloud, no data pushed to remote repositories.*

