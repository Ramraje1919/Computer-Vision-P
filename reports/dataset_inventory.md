# MOT17 Dataset Inventory and Validation Report

## Executive Summary

- **Total Sequence Folders**: 42 (21 Training, 21 Testing)
- **Unique Underlying Video Sequences**: 14 (7 Training, 7 Testing)
- **Detector Variants per Sequence**: 3 (`DPM`, `FRCNN`, `SDP`)
- **Ground Truth Status**: 100% of Training sequences (21) have valid, verified annotations; Test sequences intentionally contain no public ground truth.
- **Data Integrity**: **All 42 sequences passed 100% of structural, frame continuity, image dimension, and annotation validation checks.**

## Sequence Inventory Table

| Sequence | Variant | Split | Frames | FPS | Resolution | Duration (s) | GT Available | Active Pedestrians | Valid |
|:---|:---|:---:|---:|---:|:---:|---:|:---:|---:|:---:|
| MOT17-02 | DPM | Train | 600 | 30.0 | 1920x1080 | 20.0 | Yes | 18581 | Yes |
| MOT17-02 | FRCNN | Train | 600 | 30.0 | 1920x1080 | 20.0 | Yes | 18581 | Yes |
| MOT17-02 | SDP | Train | 600 | 30.0 | 1920x1080 | 20.0 | Yes | 18581 | Yes |
| MOT17-04 | DPM | Train | 1050 | 30.0 | 1920x1080 | 35.0 | Yes | 47557 | Yes |
| MOT17-04 | FRCNN | Train | 1050 | 30.0 | 1920x1080 | 35.0 | Yes | 47557 | Yes |
| MOT17-04 | SDP | Train | 1050 | 30.0 | 1920x1080 | 35.0 | Yes | 47557 | Yes |
| MOT17-05 | DPM | Train | 837 | 14.0 | 640x480 | 59.79 | Yes | 6917 | Yes |
| MOT17-05 | FRCNN | Train | 837 | 14.0 | 640x480 | 59.79 | Yes | 6917 | Yes |
| MOT17-05 | SDP | Train | 837 | 14.0 | 640x480 | 59.79 | Yes | 6917 | Yes |
| MOT17-09 | DPM | Train | 525 | 30.0 | 1920x1080 | 17.5 | Yes | 5325 | Yes |
| MOT17-09 | FRCNN | Train | 525 | 30.0 | 1920x1080 | 17.5 | Yes | 5325 | Yes |
| MOT17-09 | SDP | Train | 525 | 30.0 | 1920x1080 | 17.5 | Yes | 5325 | Yes |
| MOT17-10 | DPM | Train | 654 | 30.0 | 1920x1080 | 21.8 | Yes | 12839 | Yes |
| MOT17-10 | FRCNN | Train | 654 | 30.0 | 1920x1080 | 21.8 | Yes | 12839 | Yes |
| MOT17-10 | SDP | Train | 654 | 30.0 | 1920x1080 | 21.8 | Yes | 12839 | Yes |
| MOT17-11 | DPM | Train | 900 | 30.0 | 1920x1080 | 30.0 | Yes | 9436 | Yes |
| MOT17-11 | FRCNN | Train | 900 | 30.0 | 1920x1080 | 30.0 | Yes | 9436 | Yes |
| MOT17-11 | SDP | Train | 900 | 30.0 | 1920x1080 | 30.0 | Yes | 9436 | Yes |
| MOT17-13 | DPM | Train | 750 | 25.0 | 1920x1080 | 30.0 | Yes | 11642 | Yes |
| MOT17-13 | FRCNN | Train | 750 | 25.0 | 1920x1080 | 30.0 | Yes | 11642 | Yes |
| MOT17-13 | SDP | Train | 750 | 25.0 | 1920x1080 | 30.0 | Yes | 11642 | Yes |
| MOT17-01 | DPM | Test | 450 | 30.0 | 1920x1080 | 15.0 | No | None | Yes |
| MOT17-01 | FRCNN | Test | 450 | 30.0 | 1920x1080 | 15.0 | No | None | Yes |
| MOT17-01 | SDP | Test | 450 | 30.0 | 1920x1080 | 15.0 | No | None | Yes |
| MOT17-03 | DPM | Test | 1500 | 30.0 | 1920x1080 | 50.0 | No | None | Yes |
| MOT17-03 | FRCNN | Test | 1500 | 30.0 | 1920x1080 | 50.0 | No | None | Yes |
| MOT17-03 | SDP | Test | 1500 | 30.0 | 1920x1080 | 50.0 | No | None | Yes |
| MOT17-06 | DPM | Test | 1194 | 14.0 | 640x480 | 85.29 | No | None | Yes |
| MOT17-06 | FRCNN | Test | 1194 | 14.0 | 640x480 | 85.29 | No | None | Yes |
| MOT17-06 | SDP | Test | 1194 | 14.0 | 640x480 | 85.29 | No | None | Yes |
| MOT17-07 | DPM | Test | 500 | 30.0 | 1920x1080 | 16.67 | No | None | Yes |
| MOT17-07 | FRCNN | Test | 500 | 30.0 | 1920x1080 | 16.67 | No | None | Yes |
| MOT17-07 | SDP | Test | 500 | 30.0 | 1920x1080 | 16.67 | No | None | Yes |
| MOT17-08 | DPM | Test | 625 | 30.0 | 1920x1080 | 20.83 | No | None | Yes |
| MOT17-08 | FRCNN | Test | 625 | 30.0 | 1920x1080 | 20.83 | No | None | Yes |
| MOT17-08 | SDP | Test | 625 | 30.0 | 1920x1080 | 20.83 | No | None | Yes |
| MOT17-12 | DPM | Test | 900 | 30.0 | 1920x1080 | 30.0 | No | None | Yes |
| MOT17-12 | FRCNN | Test | 900 | 30.0 | 1920x1080 | 30.0 | No | None | Yes |
| MOT17-12 | SDP | Test | 900 | 30.0 | 1920x1080 | 30.0 | No | None | Yes |
| MOT17-14 | DPM | Test | 750 | 25.0 | 1920x1080 | 30.0 | No | None | Yes |
| MOT17-14 | FRCNN | Test | 750 | 25.0 | 1920x1080 | 30.0 | No | None | Yes |
| MOT17-14 | SDP | Test | 750 | 25.0 | 1920x1080 | 30.0 | No | None | Yes |

## Underlying Unique Video Sequences

The MOT17 dataset provides 3 detector variants (DPM, FRCNN, SDP) for every sequence. Image hashes confirm that **the image sequences across variants are 100% byte-identical**.

| Base Sequence | Split | Frames | Resolution | Camera Motion | Viewpoint / Scene | DPM Detections | FRCNN Detections | SDP Detections |
|:---|:---:|---:|:---:|:---:|:---|---:|---:|---:|
| MOT17-02 | Train | 600 | 1920x1080 | Static | Medium angle, Daytime indoor (mall) | 7267 | 8186 | 11639 |
| MOT17-04 | Train | 1050 | 1920x1080 | Static | High angle, Night outdoor | 39437 | 28406 | 37150 |
| MOT17-05 | Train | 837 | 640x480 | Moving | Medium angle, Daytime outdoor | 4333 | 3848 | 4767 |
| MOT17-09 | Train | 525 | 1920x1080 | Static | Low angle, Daytime outdoor | 5976 | 3049 | 3607 |
| MOT17-10 | Train | 654 | 1920x1080 | Moving | Medium angle, Night outdoor | 8832 | 9701 | 10371 |
| MOT17-11 | Train | 900 | 1920x1080 | Moving | Medium angle, Daytime indoor (mall) | 8590 | 6007 | 7509 |
| MOT17-13 | Train | 750 | 1920x1080 | Moving | Low angle (bus), Daytime outdoor | 5355 | 8442 | 7744 |
| MOT17-01 | Test | 450 | 1920x1080 | Moving | Low angle, Daytime outdoor | 3775 | 5514 | 5837 |
| MOT17-03 | Test | 1500 | 1920x1080 | Static | High angle, Night outdoor | 85854 | 65739 | 80241 |
| MOT17-06 | Test | 1194 | 640x480 | Moving | Low angle, Daytime outdoor | 7851 | 7809 | 8283 |
| MOT17-07 | Test | 500 | 1920x1080 | Moving | Medium angle, Daytime outdoor | 11309 | 9377 | 10273 |
| MOT17-08 | Test | 625 | 1920x1080 | Static | Low angle, Daytime outdoor | 10042 | 6921 | 8118 |
| MOT17-12 | Test | 900 | 1920x1080 | Moving | Low angle, Daytime outdoor | 7764 | 4726 | 5440 |
| MOT17-14 | Test | 750 | 1920x1080 | Moving | Low angle (bus), Daytime outdoor | 8781 | 10055 | 10461 |

## Ground Truth Annotation Details (Training Sequences)

| Sequence | Total GT Boxes | Active Pedestrian Boxes | Unique Pedestrian IDs | Avg Peds / Frame | Max Peds / Frame | Heavy Occlusion (<35% vis) | Full Visibility (>=90%) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 30003 | 18581 | 62 | 30.97 | 36 | 54.81% | 25.24% |
| MOT17-04 | 108005 | 47557 | 83 | 45.29 | 52 | 28.22% | 30.29% |
| MOT17-05 | 8013 | 6917 | 133 | 8.26 | 15 | 37.13% | 24.91% |
| MOT17-09 | 10411 | 5325 | 26 | 10.14 | 13 | 34.1% | 39.91% |
| MOT17-10 | 17450 | 12839 | 57 | 19.63 | 27 | 22.11% | 47.22% |
| MOT17-11 | 10617 | 9436 | 75 | 10.48 | 19 | 25.26% | 40.99% |
| MOT17-13 | 20202 | 11642 | 110 | 15.52 | 32 | 20.4% | 36.86% |

## Recommended Sequences by Tracking Challenge

Based on quantitative scene statistics and ground truth properties, the following sequences are recommended for targeted challenge analysis:

### 1. Crowded Pedestrian Tracking
- **MOT17-02**: High density with up to 36 simultaneous active pedestrians (Avg: 30.97/frame, Max: 36/frame)
- **MOT17-04**: High density with up to 52 simultaneous active pedestrians (Avg: 45.29/frame, Max: 52/frame)
- **MOT17-10**: High density with up to 27 simultaneous active pedestrians (Avg: 19.63/frame, Max: 27/frame)
- **MOT17-13**: High density with up to 32 simultaneous active pedestrians (Avg: 15.52/frame, Max: 32/frame)

### 2. Heavy Occlusion
- **MOT17-02**: 54.81% of pedestrian annotations have visibility < 35%
- **MOT17-04**: 28.22% of pedestrian annotations have visibility < 35%
- **MOT17-05**: 37.13% of pedestrian annotations have visibility < 35%
- **MOT17-09**: 34.1% of pedestrian annotations have visibility < 35%
- **MOT17-10**: 22.11% of pedestrian annotations have visibility < 35%
- **MOT17-11**: 25.26% of pedestrian annotations have visibility < 35%
- **MOT17-13**: 20.4% of pedestrian annotations have visibility < 35%

### 3. Crossing Pedestrians
- **MOT17-02**: Frequent trajectory intersections and bi-directional pedestrian flow (Camera: Static)
- **MOT17-04**: Frequent trajectory intersections and bi-directional pedestrian flow (Camera: Static)
- **MOT17-10**: Frequent trajectory intersections and bi-directional pedestrian flow (Camera: Moving)

### 4. Small Pedestrians / High Resolution
- **MOT17-02**: High elevation or wide field-of-view yielding small bounding boxes (1920x1080)
- **MOT17-04**: High elevation or wide field-of-view yielding small bounding boxes (1920x1080)
- **MOT17-13**: High elevation or wide field-of-view yielding small bounding boxes (1920x1080)

### 5. Long-Term Identity Recovery
- **MOT17-02**: Long duration (600 frames) with tracks persisting up to 600 frames across visual occluders
- **MOT17-04**: Long duration (1050 frames) with tracks persisting up to 1050 frames across visual occluders
- **MOT17-05**: Long duration (837 frames) with tracks persisting up to 394 frames across visual occluders
- **MOT17-09**: Long duration (525 frames) with tracks persisting up to 490 frames across visual occluders
- **MOT17-10**: Long duration (654 frames) with tracks persisting up to 654 frames across visual occluders
- **MOT17-11**: Long duration (900 frames) with tracks persisting up to 900 frames across visual occluders
- **MOT17-13**: Long duration (750 frames) with tracks persisting up to 396 frames across visual occluders

## Evaluation Protocols and Data Integrity Guidelines

1. **Single Representative Variant per Underlying Video**: For tracker evaluation on MOT17 public detections, evaluate each underlying sequence once per detector experiment (or report results per detector type: DPM vs FRCNN vs SDP). Never combine multiple variants into a single aggregate sequence count.
2. **Custom Pretrained Detector Evaluation**: When evaluating using an external detector (e.g. YOLOv8 / YOLO11), run detection inference only once on the underlying image frames (e.g., using the SDP directory frames) to prevent redundant computation.
3. **MOT16-03.mp4 Isolation**: Under strict project rules, `data/MOT16-03.mp4` must not be used for any quantitative evaluation, metric calculation, or parameter tuning.
4. **Zero Missing Frames**: Frame continuity verification confirmed continuous 1-indexed numbering across all 42 directories with no gaps.
