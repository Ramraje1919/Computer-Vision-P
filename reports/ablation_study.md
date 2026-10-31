# Controlled Tracker Ablation Study on MOT17

## 1. Overview and Experimental Protocol

All experiments evaluate trackers under **identical sequence images, identical cached detection inputs, and identical evaluation rules**.
Evaluation was conducted across representative sequences from the sequence-level validation split:
- **`MOT17-02-SDP`**: High-density crowded indoor mall concourse (600 frames, 18,581 ground-truth pedestrians, 54.8% heavy occlusion).
- **`MOT17-09-SDP`**: Daytime outdoor scene with clear crossing trajectories (525 frames, 5,325 ground-truth pedestrians).

## 2. Quantitative Results Comparison Table

### Results on `MOT17-02-SDP`

| Tracker Architecture | MOTA (%) | IDF1 (%) | IDP (%) | IDR (%) | IDSW | Frag | FP | FN | Precision (%) | Recall (%) | FPS |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Existing Baseline** | 41.59% | 41.79% | 61.15% | 31.74% | 301 | 876 | 808 | 9745 | 91.62% | 47.55% | 824.31 |
| **ByteTrack** | 42.35% | 45.93% | 68.76% | 34.48% | 205 | 796 | 621 | 9886 | 93.33% | 46.8% | 805.29 |
| **BoT-SORT** | 42.35% | 45.93% | 68.76% | 34.48% | 205 | 796 | 621 | 9886 | 93.33% | 46.8% | 804.86 |
| **OSNet-Assisted** | 40.86% | 29.14% | 43.3% | 21.96% | 460 | 807 | 685 | 9844 | 92.73% | 47.02% | 3.05 |
| **Final Optimized Hybrid** | 40.75% | 28.67% | 42.58% | 21.61% | 485 | 806 | 686 | 9838 | 92.72% | 47.05% | 3.33 |

### Results on `MOT17-09-SDP`

| Tracker Architecture | MOTA (%) | IDF1 (%) | IDP (%) | IDR (%) | IDSW | Frag | FP | FN | Precision (%) | Recall (%) | FPS |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Existing Baseline** | 63.0% | 52.9% | 66.73% | 43.81% | 59 | 161 | 41 | 1870 | 98.83% | 64.88% | 5913.09 |
| **ByteTrack** | 63.23% | 62.56% | 79.61% | 51.53% | 24 | 132 | 28 | 1906 | 99.19% | 64.21% | 2117.04 |
| **BoT-SORT** | 63.23% | 62.56% | 79.61% | 51.53% | 24 | 132 | 28 | 1906 | 99.19% | 64.21% | 2160.13 |
| **OSNet-Assisted** | 62.61% | 46.75% | 59.29% | 38.59% | 66 | 167 | 33 | 1892 | 99.05% | 64.47% | 7.69 |
| **Final Optimized Hybrid** | 62.72% | 42.79% | 54.25% | 35.32% | 63 | 167 | 32 | 1890 | 99.08% | 64.51% | 8.0 |

## 3. Key Algorithmic Insights and Progression Analysis

1. **Baseline (IoU + Hungarian, public SDP dets)**: With public SDP detections, the baseline achieves MOTA=41.59% on MOT17-02. Without Kalman prediction, tracks fragment quickly under occlusion (301 IDSW). Note: Stage D baseline with YOLO11n detections scored 19.93% MOTA — the jump to 41.59% is due entirely to higher-quality public detections, not tracker improvements.
2. **ByteTrack Impact (+0.76 pp MOTA, +4.14 pp IDF1 on MOT17-02)**: Two-stage BYTE association recovers low-confidence detections in a second matching round, reducing IDSW from 301 to 205 and boosting IDF1 from 41.79% to 45.93%. The Kalman filter enables motion prediction through short occlusion gaps.
3. **BoT-SORT (identical to ByteTrack without CMC)**: With Camera Motion Compensation disabled (default for CPU), BoT-SORT produces identical results to ByteTrack. CMC is computationally expensive and only beneficial for moving-camera sequences.
4. **OSNet Deep ReID — Negative Impact**: Contrary to expectations, adding OSNet-x0.25 appearance embeddings **degraded** performance. IDF1 dropped from 45.93% → 29.14% on MOT17-02, and IDSW increased from 205 → 460. The 128-D embeddings at x0.25 capacity cannot reliably discriminate similar-looking pedestrians in dense crowds.
5. **Final Optimized Hybrid — Worst Configuration**: The fully-tuned hybrid tracker achieved the lowest MOTA (40.75%) and IDF1 (28.67%) on MOT17-02. This confirms that appearance features at this model capacity are a net negative on this dataset.
