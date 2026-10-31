"""
MOT17 Tracker Evaluation Runner.
"""

import sys
import argparse
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.video_loader import MOTSequenceLoader
from src.detection.detector import DetectionCache
from src.tracking.bytetrack import ByteTracker
from src.tracking.botsort import BoTSORTTracker
from src.tracking.tracker import HybridReIDTracker
from src.reid.feature_extractor import OSNetFeatureExtractor
from src.evaluation.mot_format import write_mot_results
from src.evaluation.evaluator import evaluate_sequence

logger = get_logger("EvaluateMOT17")


def run_tracker_on_sequence(
    tracker,
    sequence_path: Path,
    cached_detections_path: Path,
    output_pred_path: Path,
    use_appearance: bool = False
):
    seq_loader = MOTSequenceLoader(sequence_path)
    dets_by_frame = DetectionCache.load(cached_detections_path)

    tracker.reset()
    all_results = []

    total = len(seq_loader)
    for frame_idx in range(1, total + 1):
        frame = seq_loader.get_frame(frame_idx) if use_appearance else None
        dets = dets_by_frame.get(frame_idx, [])
        active_tracks = tracker.update(dets, frame=frame)

        for t in active_tracks:
            x1, y1, x2, y2 = t.box
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            all_results.append((frame_idx, t.track_id, x1, y1, w, h, t.score))

    write_mot_results(all_results, output_pred_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on MOT17")
    parser.add_argument("--tracker", type=str, choices=["bytetrack", "botsort", "hybrid_reid"], default="bytetrack")
    parser.add_argument("--sequence", type=str, default="MOT17-02-SDP")
    parser.add_argument("--detector", type=str, default="public")
    parser.add_argument("--use-reid", action="store_true")
    args = parser.parse_args()

    seq_path = ROOT / "data" / "MOT17" / "train" / args.sequence
    gt_path = seq_path / "gt" / "gt.txt"
    cache_path = ROOT / "outputs" / "detections_cache" / args.detector / f"{args.sequence}.json"
    pred_path = ROOT / "outputs" / "predictions" / args.tracker / f"{args.sequence}.txt"

    # Instantiate tracker
    feat_extractor = None
    if args.use_reid:
        osnet_path = ROOT / "weights" / "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        feat_extractor = OSNetFeatureExtractor(osnet_path)

    if args.tracker == "bytetrack":
        tracker = ByteTracker(conf_high=0.50, conf_low=0.10, track_buffer=90)
    elif args.tracker == "botsort":
        tracker = BoTSORTTracker(conf_high=0.50, conf_low=0.10, track_buffer=90)
    elif args.tracker == "hybrid_reid":
        tracker = HybridReIDTracker(
            conf_high=0.50,
            conf_low=0.10,
            iou_weight=0.40,
            reid_weight=0.60,
            track_buffer=90,
            feature_extractor=feat_extractor
        )

    logger.info(f"Running {args.tracker} on {args.sequence}...")
    run_tracker_on_sequence(tracker, seq_path, cache_path, pred_path, use_appearance=args.use_reid)

    logger.info(f"Evaluating {args.sequence} metrics...")
    metrics = evaluate_sequence(gt_path, pred_path, seq_name=args.sequence)

    print("\n========== EVALUATION RESULTS ==========")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

