# MOT17 High-Accuracy Multi-Object Tracking

A modular, production-grade Multi-Object Tracking (MOT) pipeline built on the
[MOT17 benchmark](https://motchallenge.net/data/MOT17/). Implements ByteTrack,
BoT-SORT, and a Hybrid ReID tracker with full evaluation using official
MOTChallenge metrics.

---

## Results (Held-Out Benchmark, ByteTrack tuned)

| Sequence | MOTA | IDF1 | IDSW |
|---|---|---|---|
| MOT17-04-SDP (dense night) | 74.86% | 74.76% | 88 |
| MOT17-10-SDP (moving cam) | 66.05% | 47.58% | 304 |
| MOT17-11-SDP (moving cam) | 70.18% | 65.37% | 85 |
| MOT17-13-SDP (bus view) | 50.32% | 58.72% | 177 |
| **Mean** | **65.35%** | **61.61%** | 654 |

---

## Project Structure

```
PROJECT/
├── main.py                        # Unified CLI entry point
├── report.md                      # Master technical report
├── Dockerfile                     # Containerised environment
├── requirements.txt               # Python dependencies
│
├── config/
│   ├── config.yaml                # Master hyperparameters
│   └── dataset_split.yaml         # Train / val / held-out splits
│
├── src/
│   ├── detection/                 # Detector wrappers and caching
│   ├── tracking/                  # ByteTrack, BoT-SORT, HybridReID
│   ├── reid/                      # OSNet feature extraction
│   ├── counting/                  # Directional line counter
│   ├── evaluation/                # MOTChallenge metric computation
│   ├── visualization/             # Video annotation
│   └── utils/                     # Paths, logger, video loader
│
├── scripts/
│   ├── inspect_dataset.py         # Stage A: dataset validation
│   ├── generate_detections.py     # Pre-cache public detections
│   ├── run_experiments.py         # Stage G: ablation study
│   ├── tune_validation.py         # Stage H: hyperparameter search
│   ├── evaluate_held_out.py       # Stage I: held-out benchmark
│   └── evaluate_mot17.py          # Standalone evaluator
│
├── reports/
│   ├── dataset_inventory.md
│   ├── baseline_analysis.md
│   ├── architecture_design.md
│   ├── ablation_study.md
│   ├── validation_tuning_report.md
│   ├── held_out_evaluation.md
│   └── failure_analysis.md
│
├── data/
│   └── MOT17/train/<SEQ>/         # MOT17 sequences (not included)
│
├── outputs/
│   ├── detections_cache/public/   # Pre-cached public detections
│   ├── predictions/               # Per-tracker MOTChallenge prediction files
│   └── visualizations/            # Annotated MP4 videos
│
├── baseline/                      # Preserved original baseline code
├── weights/                       # OSNet model weights
├── research/                      # Algorithm comparison notes
└── tests/
    └── test_tracking_components.py  # 8 unit tests (all passing)
```

---

## Installation

**Python 3.11+ required. CPU-only — no CUDA needed.**

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `ultralytics`, `motmetrics`,
`opencv-python`, `scipy`, `numpy`, `pandas`, `torchreid`.

---

## Quick Start

### Run ByteTrack on a sequence with visualisation

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

### Pre-cache detections for all training sequences

```bash
python scripts/generate_detections.py
```

### Run ablation experiments

```bash
python scripts/run_experiments.py
```

### Run hyperparameter tuning (validation split only)

```bash
python scripts/tune_validation.py
```

### Run held-out benchmark evaluation

```bash
python scripts/evaluate_held_out.py
```

### Run unit tests

```bash
python -m pytest tests/test_tracking_components.py -v
```

---

## Tracker Architectures

### ByteTrack
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

- **Ground truth filtering**: Class 1 (Pedestrian) with `conf > 0` only.
  Non-pedestrian annotations and `conf=0` objects are treated as ignore regions.
- **Ignore region suppression**: Predictions overlapping ignore regions (IoU ≥ 0.5) are excluded.
- **IoU threshold**: 0.50 for match/no-match decisions (MOTChallenge standard).
- **Metrics**: MOTA, IDF1, IDP, IDR, IDSW, Frag, FP, FN, Precision, Recall, MT, ML.

---

## Data Split

| Split | Sequences | Purpose |
|---|---|---|
| Validation | MOT17-02, 05, 09 (SDP) | Hyperparameter tuning only |
| Held-out | MOT17-04, 10, 11, 13 (SDP) | Final benchmark |

`data/MOT16-03.mp4` is excluded from all metric computation.

---

## Docker

```bash
docker build -t mot17-tracker .
docker run --rm -v $(pwd)/data:/app/data mot17-tracker \
    python main.py --sequence MOT17-09-SDP --tracker bytetrack
```

---

## Reproducing Results

1. Place MOT17 sequences under `data/MOT17/train/<SEQ_NAME>/`
2. Run `python scripts/generate_detections.py` to cache detections
3. Run `python scripts/evaluate_held_out.py` for the final benchmark
4. See `reports/held_out_evaluation.md` for results

All experiments are reproducible: no randomness in ByteTrack or BoT-SORT (deterministic IoU matching + Kalman prediction). Unit tests verify all core components.

---

## Reports

| Report | Description |
|---|---|
| [`report.md`](report.md) | Master technical report |
| [`reports/ablation_study.md`](reports/ablation_study.md) | Algorithm comparison |
| [`reports/validation_tuning_report.md`](reports/validation_tuning_report.md) | Hyperparameter optimisation |
| [`reports/held_out_evaluation.md`](reports/held_out_evaluation.md) | Final benchmark results |
| [`reports/failure_analysis.md`](reports/failure_analysis.md) | Failure modes and analysis |
| [`research/algorithm_comparison.md`](research/algorithm_comparison.md) | Algorithm literature review |
