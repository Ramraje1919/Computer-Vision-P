# Multi-Object Tracking (MOT) Algorithm Research and Comparative Analysis

## 1. Executive Summary

This document presents an in-depth research review of leading multi-object tracking (MOT) paradigms to inform the architectural design and implementation for our MOT17 tracking system. We investigate seven foundational tracking frameworks, algorithms, and evaluation systems:
1. **ByteTrack** (Zhang et al., ECCV 2022)
2. **BoT-SORT** (Aharon et al., arXiv 2022)
3. **StrongSORT** (Du et al., IEEE TMM 2023)
4. **DeepSORT** (Wojke et al., ICIP 2017)
5. **OC-SORT** (Cao et al., NeurIPS 2022)
6. **BoxMOT** (Brostrom, 2023)
7. **TrackEval** (Luiten et al., IJCV 2021)

> [!NOTE]
> **Metric Attribution Notice**: All quantitative metrics listed in this research report reflect **published external benchmark results on the official MOT17 test set** from their respective original peer-reviewed papers or official repositories. They do NOT represent experimental results from this project. Our project's quantitative metrics will be derived exclusively from real evaluation against the downloaded MOT17 ground truth in subsequent stages.

---

## 2. Comparative Algorithm Matrix

The following table summarizes the mathematical formulations, components, and published MOT17 test set performances across the evaluated algorithms:

| Algorithm | Motion Model | Appearance Model | Occlusion Recovery Strategy | Published MOTA (%) | Published IDF1 (%) | Published IDSW | Computational Complexity | Primary Failure Mode |
|:---|:---|:---|:---|---:|---:|---:|:---|:---|
| **DeepSORT** | Standard KF $[x,y,a,h,\dot{x},\dot{y},\dot{a},\dot{h}]$ | CNN (128-d) + cosine metric | Matching cascade across age | 64.1 | 62.2 | 781 | Medium (~20 FPS) | Drops low-score detections; aspect-ratio drift |
| **ByteTrack** | Standard KF $[x,y,a,h,\dot{x},\dot{y},\dot{a},\dot{h}]$ | Motion-only (No ReID) | Two-stage BYTE association ($D_{high} \rightarrow D_{low}$) | **80.3** | 77.3 | 896 | **Very Low (~30+ FPS)** | Crossing paths with identical trajectories; long occlusions |
| **BoT-SORT** | Modified KF $[x,y,w,h,\dot{x},\dot{y},\dot{w},\dot{h}]$ + CMC (GMC) | OSNet / FastReID + EMA Gallery | BYTE two-stage + Appearance/IoU cost fusion + CMC | **80.5** | **80.2** | 1212 | Medium (~15-22 FPS) | Computational cost of GMC on high-res moving video |
| **StrongSORT** | NSA Kalman Filter (Adaptive $R$) + ECC CMC | Deep ReID + EMA bank + AFLink | Global link graph (AFLink) + GSI interpolation | 79.6 | 79.5 | 1194 | High (~10-15 FPS) | High latency; AFLink requires offline/sliding window |
| **OC-SORT** | Standard KF + Observation-Centric Re-update | Motion-only (Direction consistency OCM) | Observation-Centric Recovery (OCR) + Back-propagation | 78.0 | 77.5 | **489** | Low (~28 FPS) | Visually indistinguishable crossing pedestrians |
| **BoxMOT** | Framework | Multi-ReID (OSNet, CLIP, FastReID) | Wrapper implementing above algos | Config | Config | Config | Varies by backbone | Architectural abstraction overhead |
| **TrackEval** | N/A (Evaluator) | N/A (Evaluator) | N/A (Evaluator) | Metric tool | Metric tool | Metric tool | N/A | Evaluates HOTA, DetA, AssA, MOTA, IDF1 |

---

## 3. Deep Algorithmic Analysis

### 3.1 ByteTrack: Multi-Object Tracking by Associating Every Detection Box

#### Architecture & Mathematical Formulation
Traditional tracking-by-detection methods filter detections with a high confidence threshold (e.g., $\tau_{det} = 0.6$), entirely discarding low-confidence bounding boxes. However, occluded, blurred, or distant pedestrians frequently generate detection confidences in the range $[0.1, 0.6]$. ByteTrack addresses this via its **BYTE association pipeline**:

1. **Split Detections**: Every frame detection set $D$ is partitioned into high-confidence detections $D_{high} = \{d \in D \mid \text{score}(d) \ge \tau_{high}\}$ and low-confidence detections $D_{low} = \{d \in D \mid \tau_{low} \le \text{score}(d) < \tau_{high}\}$.
2. **First Association Stage (High-Score)**:
   - Tracks $T$ are updated via Kalman filter state prediction: $\hat{x}_{k|k-1} = F x_{k-1}$.
   - Match $T$ against $D_{high}$ using IoU distance $C_{IoU} = 1 - \text{IoU}(\hat{B}_T, B_{D_{high}})$.
   - Solve assignment via the Hungarian (Munkres) or Jonker-Volgenant algorithm.
   - Matched pairs update tracks: $T_{matched}$.
3. **Second Association Stage (Low-Score Occlusion Recovery)**:
   - Unmatched tracks $T_{remain}$ from the first stage are matched against $D_{low}$ using IoU distance.
   - Because $D_{low}$ boxes are occluded/blurred pedestrians, matching them against already-tracked hypotheses recovers objects emerging from occlusion without hallucinating false positive tracks.
   - Unmatched tracks remaining after Stage 2 are marked `LOST`.
   - Remaining unmatched $D_{low}$ detections are discarded (never used to initialize new tracks).
4. **Track Initialization**:
   - Only unmatched high-score detections $D_{remain\_high}$ initialize new track candidates. Candidates must be matched in two consecutive frames before promotion to `TRACKED`.

#### Strengths
- Unprecedented MOTA boost on MOT17 by eliminating false negatives on partially occluded targets.
- Minimal compute overhead (pure matrix association).

#### Limitations
- Motion-only ByteTrack fails during path intersections ("crossing people") and long-term occlusions (> 30 frames) where Kalman filter predictions diverge.

---

### 3.2 BoT-SORT: Robust Multi-Person Tracking by Integrating Motion and Appearance

BoT-SORT introduced three decisive tracking enhancements:

#### 1. Camera Motion Compensation (CMC / GMC)
In moving-camera sequences (e.g., MOT17-05, MOT17-10, MOT17-11, MOT17-13), camera ego-motion violates the constant-velocity assumption of the Kalman Filter. BoT-SORT estimates an affine or homography transformation matrix $H \in \mathbb{R}^{2 \times 3}$ between consecutive frames $I_{k-1}$ and $I_k$ using sparse optical flow or ORB feature matching with RANSAC.
The Kalman state $\hat{x}_{k|k-1}$ and covariance $P_{k|k-1}$ are transformed before association:
$$\hat{x}' = A_H \hat{x}_{k|k-1}, \quad P' = A_H P_{k|k-1} A_H^T$$

#### 2. Width-Height Kalman Filter State Formulation
Standard SORT and ByteTrack model the bounding box as $[x, y, a, h]$ where $a = w/h$ is the aspect ratio. However, $a$ fluctuates drastically under noisy detections, leading to unstable bounding box scaling. BoT-SORT reformulates the state vector directly to:
$$x = [x_c, y_c, w, h, \dot{x}_c, \dot{y}_c, \dot{w}, \dot{h}]^T$$
Directly tracking width and height significantly stabilizes box dimensions during occlusion.

#### 3. Appearance-Feature and IoU Fusion
BoT-SORT integrates appearance features within the BYTE paradigm:
- Computes cosine distance matrix $D_{emb} = 1 - \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}$.
- Fuses appearance distance with IoU distance:
  $$C_{fuse} = \alpha D_{emb} + (1 - \alpha) C_{IoU}$$
- Employs strict motion/spatial gating: if IoU is zero or Mahalanobis distance exceeds threshold $\gamma$, appearance matching is rejected to avoid distant impostor matches.
- Track appearance is updated with an Exponential Moving Average (EMA) gallery:
  $$e_k^i = \beta e_{k-1}^i + (1 - \beta) f_k^i, \quad e_k^i \leftarrow \frac{e_k^i}{\|e_k^i\|_2}$$

---

### 3.3 StrongSORT: DeepSORT Revisited

StrongSORT systematically modernizes DeepSORT through three components:
1. **Noise Scale Adaptive (NSA) Kalman Filter**:
   Instead of a constant measurement noise covariance $R_k$, the noise is scaled inversely with detection confidence $c_k$:
   $$\tilde{R}_k = (1 - c_k) R_k$$
   Detections with low confidence have high noise uncertainty, preventing the Kalman filter state from overfitting to degraded bounding boxes.
2. **Appearance-Free Link (AFLink)**:
   A lightweight spatial-temporal convolutional network that operates as a global post-processor, predicting whether two separated tracklets belong to the same identity based purely on temporal intervals and bounding box motion.
3. **Gaussian-smoothed Interpolation (GSI)**:
   Interpolates missing bounding boxes during occlusions using Gaussian process regression rather than simple linear interpolation.

---

### 3.4 OC-SORT: Observation-Centric SORT

OC-SORT observed that the primary cause of ID switches in motion-based tracking is the accumulation of error in the Kalman filter during occlusion. It introduces:
1. **Observation-Centric Online Re-update (OOR)**:
   When a lost track is matched with a new detection after $t$ frames of occlusion, it performs a virtual backward pass, re-estimating the trajectory and updating the Kalman filter's velocity and covariance state using actual endpoints.
2. **Observation-Centric Momentum (OCM)**:
   Penalizes association costs where the implied velocity vector deviates sharply from the target's historical momentum direction:
   $$C_{OCM} = C_{IoU} + \lambda \cdot (1 - \cos(\theta_{motion}, \theta_{pair}))$$
3. **Observation-Centric Recovery (OCR)**:
   For tracks that remain lost, OCR checks association against the last reliable observation position rather than the drifted Kalman extrapolation.

---

### 3.5 Deep Person Re-Identification: OSNet (Omni-Scale Network)

Person Re-Identification (ReID) provides visual appearance invariance across camera angle shifts, scale variations, and occlusion:
- **Omni-Scale Feature Learning**: OSNet (Zhou et al., ICCV 2019) utilizes depth-wise separable convolutional residual blocks with multiple receptive field streams ($\{1\times 1, 3\times 3, 5\times 5, \dots\}$), capturing both fine-grained local patterns (shoes, bags, logo patterns) and whole-body structural features.
- **Lightweight Architecture**: OSNet (e.g. `osnet_x0_25`, `osnet_x1_0`, `osnet_ibn_x1_0`) has a very small parameter footprint (~2.2M parameters for x1_0), enabling fast feature extraction on CPU and GPU.
- **IBN (Instance-Batch Normalization)**: Enhances cross-domain generalization by decoupling illumination and domain variations from semantic identity features.

---

### 3.6 Evaluation Methodology: TrackEval Benchmark Standards

`TrackEval` (Jonathon Luiten et al.) is the unified official evaluation engine for MOTChallenge. Key metrics evaluated include:

1. **HOTA (Higher Order Tracking Accuracy)**:
   $$HOTA_\alpha = \sqrt{\frac{\sum_{c \in C} \mathcal{A}(c)}{|TP| + |FN| + |FP|}}$$
   Balances detection accuracy ($DetA$) and association accuracy ($AssA$) cleanly without favoring either recall or precision disproportionately.
2. **MOTA (Multi-Object Tracking Accuracy)**:
   $$MOTA = 1 - \frac{\sum_t (FN_t + FP_t + IDSW_t)}{\sum_t GT_t}$$
   Heavily influenced by detection quality ($FP$ and $FN$); heavily penalizes false detections.
3. **IDF1 (Identification F1-Score)**:
   $$IDF1 = \frac{2 IDTP}{2 IDTP + IDFP + IDFN}$$
   Measures how accurately identities are preserved over long horizons; directly penalizes ID switches and fragmented tracks.
4. **ID Switches ($IDSW$)**:
   The count of instances where a tracked target changes its assigned hypothesis ID.
5. **Class & Visibility Filtering**:
   TrackEval adheres strictly to MOTChallenge ground truth conventions: only Class 1 (`Pedestrian`) with `conf=1` is evaluated. Classes 2 (Person on vehicle), 7 (Static person), 8 (Distractor), etc. with `conf=0` are treated as non-target ignore regions. Detections falling inside ignore regions do not incur False Positive penalties.

---

## 4. Key Takeaways and Architectural Blueprint for Our System

From our research across these algorithms, the optimal architecture for our MOT17 tracking system combines the highest-performing mechanisms:

1. **Detection Foundation**: High-quality pedestrian detection via state-of-the-art detector (e.g. YOLOv8 / YOLO11) cached offline to eliminate detection bottlenecks.
2. **Motion Estimation**: BoT-SORT-style 8-state Kalman Filter $[x, y, w, h, \dot{x}, \dot{y}, \dot{w}, \dot{h}]$ to avoid aspect-ratio instability.
3. **Camera Motion Compensation**: Homography-based or affine feature matching on keyframes to stabilize track trajectories in moving-camera sequences (MOT17-05, 10, 11, 13).
4. **Association Strategy**: Two-stage BYTE association with appearance fusion:
   - Stage 1: Match active tracks with $D_{high}$ using a fused cost matrix: $C = \lambda C_{IoU} + (1 - \lambda) C_{emb}$ subject to spatial gating.
   - Stage 2: Match remaining tracks with $D_{low}$ using IoU/motion to recover occluded targets.
5. **Appearance Memory & Occlusion Recovery**:
   - OSNet deep feature extractor extracting normalized 512-d embeddings.
   - EMA-updated feature gallery + historical exemplar memory for long occlusions.
   - Separate candidate pool for re-entry and long-term identity recovery.
6. **Strict Standardized Evaluation**: Integration of TrackEval directly on MOT17 ground truth.

---
*Report compiled for MOT17 Research Phase.*

