# Robust Multi-Object Tracking in Crowded, Occluded Video

## Approach Summary

This project builds a **multi-object pedestrian tracking pipeline** that detects, tracks, and counts people across video frames — maintaining persistent identities through occlusions, crossings, and scale changes. The system uses **YOLO11n** for detection and **ByteTrack** (BYTE two-stage association) for tracking, chosen after researching and experimentally comparing four tracker architectures (Baseline IoU, ByteTrack, BoT-SORT, and a Hybrid ReID tracker with OSNet appearance embeddings). ByteTrack was selected as the final tracker because it achieved the best MOTA/IDF1 on our validation split while being the simplest and fastest — its key insight is that low-confidence detections, normally discarded, can recover occluded tracks when matched to predicted positions in a second association stage. The pipeline also includes a **directional line counter** (IN/OUT counting by track ID with re-crossing prevention), **Kalman filter** motion prediction through occlusion gaps, and a **track lifecycle FSM** (Tentative → Confirmed → Lost → Dead) that prevents spurious detections from creating false tracks.

Evaluated on MOT17 with official ground truth using a strict **validation/held-out split protocol** — hyperparameters tuned only on the validation split, final numbers reported on the held-out split. Achieves **Mean MOTA = 65.35%** and **Mean IDF1 = 61.61%** on 4 held-out sequences.

---

## Alternatives Considered & Design Decisions

This section documents the R&D process — approaches researched, tested, and the reasoning behind each decision.

### Tracker Selection

| Approach | Considered? | Tested? | Outcome |
|---|---|---|---|
| **SORT** (Bewley 2016) | ✅ Researched | As baseline | Simple IoU + Kalman. No low-conf recovery → high FN in occlusion. Our baseline implementation. |
| **DeepSORT** (Wojke 2017) | ✅ Researched | ❌ | Adds CNN appearance features to SORT. Cascade matching adds complexity. Superseded by ByteTrack/StrongSORT in benchmarks. Rejected in favor of testing the appearance idea within our HybridReID tracker. |
| **ByteTrack** (Zhang 2022) | ✅ Researched | ✅ **Selected** | Two-stage BYTE association recovers occluded tracks using low-confidence detections. Simple, fast, highest MOTA/IDF1 in our experiments. No appearance features needed. |
| **BoT-SORT** (Aharon 2022) | ✅ Researched | ✅ Tested | ByteTrack + Camera Motion Compensation (CMC). Produced identical results when CMC disabled. CMC helpful for moving-camera sequences but expensive on CPU (~3× slower). Not worth the cost for static-camera-dominant use case. |
| **StrongSORT** (Du 2023) | ✅ Researched | ❌ | Extends DeepSORT with stronger ReID (BoT + OSNet). Promising but requires GPU for real-time ReID inference. Concepts tested through our HybridReID tracker. |
| **OC-SORT** (Cao 2023) | ✅ Researched | ❌ | Observation-centric Kalman recovery. Interesting virtual trajectory idea, but ByteTrack's two-stage approach already addresses the core problem (occluded track recovery) more simply. |
| **HybridReID (ours)** | — | ✅ Tested | ByteTrack + OSNet-x0.25 appearance embeddings fused into cost matrix (40% IoU / 60% appearance). **Result: IDF1 dropped from 45.93% → 29.14%** on dense sequences. Appearance features unreliable at x0.25 capacity on CPU with homogeneous-looking pedestrians. Rejected. |

**Key finding**: On CPU with public detections, **motion-only ByteTrack outperforms appearance-augmented trackers**. The OSNet-x0.25 embeddings at 128-D cannot reliably discriminate similar-looking pedestrians in dense crowds, and the fused cost matrix amplifies appearance errors. Appearance features should be subordinate to (not equal to) motion cues, or require a stronger model on GPU. See [reports/ablation_study.md](reports/ablation_study.md) for full ablation with metrics.

### Detector Selection

| Option | Decision | Reason |
|---|---|---|
| MOT17 public detections (SDP) | **Used for benchmarking** | Enables fair comparison against published results. High precision, moderate recall. |
| YOLO11n (pretrained, COCO) | **Used for MOT16-03 video** | Real-time inference (~12 FPS on CPU). No public detections available for raw MP4 video. |
| Fine-tuned YOLO on MOT17 | Not pursued | Limited training data; risk of overfitting to 7 unique scenes. Public detections already provide a strong baseline. |

### Occlusion Handling Strategy

We use three complementary mechanisms:

1. **BYTE two-stage association**: Detections with confidence in [0.10, 0.50) — normally discarded — are matched to tracks that failed the high-confidence stage. This recovers partially-occluded pedestrians whose detector confidence dropped below the primary threshold.

2. **Kalman filter prediction**: During occlusion, the 8-state constant-velocity Kalman filter continues predicting the track's position for up to `track_buffer=90` frames (~3 seconds at 30 FPS). When the person re-emerges, the predicted position enables correct re-association via IoU matching.

3. **Track lifecycle FSM**: New detections enter as `Tentative` (requires 2 consecutive matches to become `Confirmed`, preventing false tracks from noise). Lost tracks remain in `Lost` state for up to `track_buffer` frames before becoming `Dead`, giving occluded persons time to re-appear.

We also implemented **OSNet appearance embeddings** (feature gallery with cosine distance matching) as a fourth mechanism, but ablation showed it hurt performance on our data — see "Alternatives Considered" above.

---

## Results

### Held-Out Benchmark (ByteTrack, tuned)

| Sequence | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|
| MOT17-04-SDP (dense night) | 74.86% | 74.76% | 88 | 105 | 11,762 |
| MOT17-10-SDP (moving cam) | 66.05% | 47.58% | 304 | 528 | 3,527 |
| MOT17-11-SDP (moving cam) | 70.18% | 65.37% | 85 | 192 | 2,537 |
| MOT17-13-SDP (bus view) | 50.32% | 58.72% | 177 | 486 | 5,121 |
| **Mean (arithmetic, per-sequence)** | **65.35%** | **61.61%** | **654** | **1,311** | **22,947** |

> **Note**: Mean is the arithmetic average of per-sequence metrics; sequences with more frames/objects are not weighted more heavily. These are MOT17 *training* sequences reserved as a held-out evaluation set — not the official MOT17 test split (which has no public ground truth).

### MOT16-03 Demo (Full Video)

| Metric | Value |
|---|---|
| Video | `outputs/MOT16-03/MOT16-03_tracked.mp4` (1,500 frames, 960×540) |
| Unique track IDs | 92 |
| Max simultaneous tracks | 21 |
| Line-crossing counts | IN=15, OUT=28, Total=43 |
| Processing speed | 12.1 FPS (CPU) |

### Ablation: IoU-Only vs IoU+Appearance (on MOT17-02-SDP)

| Tracker | MOTA | IDF1 | IDSW |
|---|---|---|---|
| Baseline (IoU-only, no Kalman) | 41.59% | 41.79% | 301 |
| ByteTrack (IoU + Kalman + BYTE) | **42.35%** | **45.93%** | **205** |
| HybridReID (IoU + Kalman + OSNet) | 40.86% | 29.14% | 460 |

---

## Sample Output Video

The annotated demo video with bounding boxes, persistent track IDs, motion trails, and live IN/OUT counting overlay:

- **Full MOT16-03**: [`outputs/MOT16-03/MOT16-03_tracked.mp4`](outputs/MOT16-03/MOT16-03_tracked.mp4)
- **MOT17-09-SDP** (150 frames): [`outputs/visualizations/MOT17-09-SDP_bytetrack.mp4`](outputs/visualizations/MOT17-09-SDP_bytetrack.mp4)
- **MOT17-11-SDP** (150 frames): [`outputs/visualizations/MOT17-11-SDP_bytetrack.mp4`](outputs/visualizations/MOT17-11-SDP_bytetrack.mp4)

---

## How to Run (End-to-End)

### Installation

**Python 3.11+ required. CPU-only — no CUDA needed.**

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `ultralytics`, `motmetrics`,
`opencv-python`, `scipy`, `numpy`, `pandas`.

> **Note**: `torchreid` is required for the OSNet ReID module. Install via:
> `pip install torchreid` (or `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git` if PyPI version is unavailable). Not needed if only using ByteTrack (the default, recommended tracker).

### Run tracking with visualisation on MOT16-03.mp4

```bash
python scripts/track_mot16_03.py
```

This runs YOLO11n detection + ByteTrack tracking + line counting on the full video and saves the annotated output to `outputs/MOT16-03/`.

### Run on any MOT17 sequence

```bash
python main.py --sequence MOT17-09-SDP --tracker bytetrack --detector public --visualize --count
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--sequence` | `MOT17-02-SDP` | Sequence name (must exist in `data/MOT17/train/`) |
| `--tracker` | `bytetrack` | `bytetrack`, `botsort`, or `hybrid_reid` |
| `--detector` | `public` | `public` (MOT17 gt-format dets) or `yolo` (YOLO11n) |
| `--use-reid` | off | Enable OSNet deep appearance embeddings |
| `--visualize` | off | Write annotated MP4 to `outputs/visualizations/` |
| `--count` | off | Enable virtual line directional counter |
| `--max-frames` | all | Limit frames processed (for quick tests) |

### Reproduce the benchmark results

```bash
python scripts/generate_detections.py        # Pre-cache public detections
python scripts/run_experiments.py             # Ablation study
python scripts/tune_validation.py             # Hyperparameter search (validation only)
python scripts/evaluate_held_out.py           # Final held-out benchmark
python -m pytest tests/test_tracking_components.py -v   # Unit tests (8/8 passing)
```

### Docker

```bash
docker build -t mot17-tracker .
docker run --rm -v $(pwd)/data:/app/data mot17-tracker \
    python main.py --sequence MOT17-09-SDP --tracker bytetrack
```

---

## Project Structure

```
PROJECT/
├── main.py                        # Unified CLI entry point
├── report.md                      # Master technical report (design decisions + failure analysis)
├── README.md                      # This file
├── Dockerfile                     # Containerised environment
├── requirements.txt               # Python dependencies
│
├── config/
│   ├── config.yaml                # Master hyperparameters
│   └── dataset_split.yaml         # Train / val / held-out splits
│
├── src/
│   ├── detection/                 # Detector wrappers and caching
│   │   ├── detector.py            # YOLODetector, PublicDetectionLoader, DetectionCache
│   │   ├── preprocessing.py       # Frame normalisation
│   │   └── postprocessing.py      # NMS, box format conversion
│   ├── tracking/                  # Core tracking algorithms
│   │   ├── kalman_filter.py       # 8-state BoT-SORT Kalman filter
│   │   ├── hungarian_matcher.py   # Linear assignment (scipy)
│   │   ├── association.py         # IoU distance, fused cost, spatial gating
│   │   ├── track.py               # Track entity with FSM lifecycle
│   │   ├── track_manager.py       # Track pool management
│   │   ├── bytetrack.py           # ByteTrack tracker
│   │   ├── botsort.py             # BoT-SORT tracker (with optional CMC)
│   │   └── tracker.py             # HybridReIDTracker
│   ├── reid/                      # Deep appearance re-identification
│   │   ├── osnet.py               # Direct OSNet loader
│   │   ├── feature_extractor.py   # Inference wrapper
│   │   ├── feature_gallery.py     # Multi-sample exemplar memory
│   │   └── reidentifier.py        # ReID decision module
│   ├── counting/                  # Directional person counting
│   │   └── line_counter.py        # Virtual line counter (IN/OUT + cooldown)
│   ├── evaluation/                # Metric computation
│   │   ├── evaluator.py           # motmetrics-based MOT evaluator
│   │   └── mot_format.py          # MOTChallenge gt/pred I/O
│   ├── visualization/             # Video annotation
│   │   └── visualizer.py          # Boxes, IDs, trails, dashboard, counter overlay
│   └── utils/
│       ├── paths.py, logger.py, video_loader.py
│
├── scripts/
│   ├── inspect_dataset.py         # Dataset validation (42 sequences)
│   ├── generate_detections.py     # Pre-cache public detections
│   ├── run_experiments.py         # Ablation study (4 trackers)
│   ├── tune_validation.py         # Hyperparameter grid search
│   ├── evaluate_held_out.py       # Held-out benchmark
│   ├── evaluate_mot17.py          # Standalone evaluator
│   └── track_mot16_03.py          # MOT16-03 full-video demo
│
├── reports/                       # Generated reports
│   ├── ablation_study.md          # Algorithm comparison with metrics
│   ├── failure_analysis.md        # Failure modes analysis
│   ├── held_out_evaluation.md     # Final benchmark
│   ├── validation_tuning_report.md
│   ├── baseline_analysis.md
│   ├── architecture_design.md
│   └── dataset_inventory.md
│
├── outputs/
│   ├── MOT16-03/                  # MOT16-03 annotated video + report
│   ├── visualizations/            # Additional annotated MP4 videos
│   ├── detections_cache/          # Pre-cached detections
│   └── predictions/               # MOTChallenge prediction files
│
├── baseline/                      # Preserved original baseline code
├── weights/                       # OSNet model weights
├── research/                      # Algorithm literature review
│   └── algorithm_comparison.md
├── tests/
│   └── test_tracking_components.py  # 8 unit tests (all passing)
└── data/
    ├── MOT16-03.mp4               # Demo video (MOT16-03 sequence)
    └── MOT17/train/<SEQ>/         # MOT17 benchmark sequences
```

---

## Tracker Architectures

### ByteTrack (Selected)

Two-stage BYTE association: high-confidence detections matched first (IoU),
then low-confidence detections matched to remaining tracks. Kalman filter
(8-state BoT-SORT formulation) for motion prediction.

**Best for**: Dense scenes, static cameras. Fastest and most reliable.

### BoT-SORT

ByteTrack + optional Camera Motion Compensation (CMC / GMC).
CMC disabled by default (CPU cost); enable for moving-camera sequences.

**Best for**: Moving-camera sequences when CMC is enabled.

### HybridReIDTracker

ByteTrack + OSNet appearance features fused into the cost matrix
(`iou_weight=0.40, reid_weight=0.60`). Adds `osnet_x0_25_msmt17` embeddings.

**Note**: Per ablation study, ReID degrades IDF1 on dense sequences with
homogeneous appearance. Use ByteTrack unless GPU inference is available.

---

## Evaluation Protocol

- **Ground truth filtering**: Class 1 (Pedestrian) with `conf > 0` only. Non-pedestrian annotations and `conf=0` objects are treated as ignore regions.
- **Ignore region suppression**: Predictions overlapping ignore regions (IoU ≥ 0.5) are excluded from FP count.
- **IoU threshold**: 0.50 for match/no-match decisions (MOTChallenge standard).
- **Metrics**: MOTA, IDF1, IDP, IDR, IDSW, Frag, FP, FN, Precision, Recall, MT, ML.
- **Data split**: Validation (MOT17-02, 05, 09) for tuning; Held-out (MOT17-04, 10, 11, 13) for final evaluation. No cross-contamination.

---

## Known Limitations & What I'd Do With More Time

### Current Limitations

1. **Detection recall is the bottleneck**: 93% of all errors (by count) are false negatives — pedestrians the detector never detects. The tracker cannot track what it never sees. Improving the detector would have a dramatically larger impact than any tracker improvement.

2. **No Camera Motion Compensation**: ByteTrack uses pixel-space Kalman prediction, which fails when the camera pans/tilts (all pedestrians shift simultaneously). This causes high IDSW on MOT17-10 (304 switches) and MOT17-11. BoT-SORT includes CMC but it's too slow on CPU.

3. **Long occlusions (>3 seconds)**: `track_buffer=90` frames means any occlusion longer than 3 seconds at 30 FPS results in track ID loss. Re-entering pedestrians get new IDs.

4. **ReID unreliable on CPU**: OSNet-x0.25 at 128-D embeddings cannot reliably discriminate similar-looking pedestrians in dense crowds. The appearance cost actually hurts performance (ablation: IDF1 dropped from 45.93% → 29.14%).

5. **CPU-only FPS**: Processing at ~12 FPS on CPU is below real-time (15 FPS source video). GPU would enable real-time processing and stronger ReID models.

### With More Time

| Priority | Improvement | Expected Impact |
|---|---|---|
| High | Fine-tune a stronger detector (YOLOv8-X or RT-DETR) on MOT17 | +15–25 pp MOTA (reduce FN) |
| High | Enable CMC/GMC in BoT-SORT with GPU | −30–50% IDSW on moving-camera sequences |
| Medium | Adaptive `track_buffer` per sequence type | −10–20% IDSW |
| Medium | Velocity-gated ReID: apply appearance only when IoU < 0.3 | +3–5 pp IDF1 on crossing scenarios |
| Low | Multi-detector ensemble (DPM + FRCNN + SDP via NMS) | +5–10 pp recall |
| Low | Online learning for appearance model adaptation | Better ReID discrimination over time |

---

## Reports

| Report | Description |
|---|---|
| [`report.md`](report.md) | Master technical report (all stages) |
| [`reports/ablation_study.md`](reports/ablation_study.md) | 4-tracker comparison with metrics |
| [`reports/validation_tuning_report.md`](reports/validation_tuning_report.md) | Hyperparameter grid search results |
| [`reports/held_out_evaluation.md`](reports/held_out_evaluation.md) | Final held-out benchmark |
| [`reports/failure_analysis.md`](reports/failure_analysis.md) | Detailed failure mode analysis |
| [`research/algorithm_comparison.md`](research/algorithm_comparison.md) | Algorithm literature review |
| [`outputs/MOT16-03/tracking_report.md`](outputs/MOT16-03/tracking_report.md) | MOT16-03 demo report |

