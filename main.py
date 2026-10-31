"""
Main Command-Line Interface for the MOT17 High-Accuracy Tracking Pipeline.
"""

import sys
import argparse
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.video_loader import MOTSequenceLoader
from src.detection.detector import YOLODetector, PublicDetectionLoader, DetectionCache
from src.tracking.bytetrack import ByteTracker
from src.tracking.botsort import BoTSORTTracker
from src.tracking.tracker import HybridReIDTracker
from src.reid.feature_extractor import OSNetFeatureExtractor
from src.counting.line_counter import DirectionalLineCounter
from src.evaluation.evaluator import evaluate_sequence
from src.evaluation.mot_format import write_mot_results
from src.visualization.visualizer import MOTVisualizer, VideoWriterWrapper

logger = get_logger("MainApp")


def main():
    parser = argparse.ArgumentParser(description="High-Accuracy Multi-Object Tracking Pipeline on MOT17")
    parser.add_argument("--sequence", type=str, default="MOT17-02-SDP", help="Sequence name (e.g. MOT17-02-SDP)")
    parser.add_argument("--tracker", type=str, choices=["bytetrack", "botsort", "hybrid_reid"], default="bytetrack")
    parser.add_argument("--detector", type=str, choices=["public", "yolo"], default="public")
    parser.add_argument("--use-reid", action="store_true", help="Enable OSNet deep appearance embeddings")
    parser.add_argument("--visualize", action="store_true", help="Generate annotated video with tracks and counters")
    parser.add_argument("--count", action="store_true", help="Enable virtual line directional person counter")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()

    seq_path = ROOT / "data" / "MOT17" / "train" / args.sequence
    if not seq_path.exists():
        seq_path = ROOT / "data" / "MOT17" / "test" / args.sequence
    if not seq_path.exists():
        logger.error(f"Sequence not found: {args.sequence}")
        sys.exit(1)

    logger.info(f"Initializing tracking on sequence: {args.sequence}")
    seq_loader = MOTSequenceLoader(seq_path)
    total_frames = args.max_frames or len(seq_loader)

    # 1. Detection loading / caching
    cache_dir = ROOT / "outputs" / "detections_cache" / args.detector
    cache_file = cache_dir / f"{args.sequence}.json"

    if cache_file.exists():
        logger.info(f"Loading cached detections from: {cache_file}")
        dets_by_frame = DetectionCache.load(cache_file)
    else:
        logger.info(f"Parsing detections for {args.sequence}...")
        if args.detector == "public":
            loader = PublicDetectionLoader(seq_path / "det" / "det.txt")
            dets_by_frame = loader.detections_by_frame
        else:
            yolo = YOLODetector(weights_path="yolo11n.pt")
            dets_by_frame = {}
            for f_idx, frame in seq_loader.iter_frames():
                if args.max_frames and f_idx > args.max_frames:
                    break
                dets_by_frame[f_idx] = yolo.predict(frame)
        DetectionCache.save(dets_by_frame, cache_file)

    # 2. Tracker initialization
    feat_extractor = None
    if args.use_reid or args.tracker == "hybrid_reid":
        weights_path = ROOT / "weights" / "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        feat_extractor = OSNetFeatureExtractor(weights_path)

    if args.tracker == "bytetrack":
        tracker = ByteTracker(conf_high=0.50, conf_low=0.10, track_buffer=90)
    elif args.tracker == "botsort":
        tracker = BoTSORTTracker(conf_high=0.50, conf_low=0.10, track_buffer=90)
    else:
        tracker = HybridReIDTracker(
            conf_high=0.50,
            conf_low=0.10,
            iou_weight=0.40,
            reid_weight=0.60,
            track_buffer=90,
            feature_extractor=feat_extractor
        )

    # 3. Optional directional line counter
    counter = None
    if args.count:
        y_line = int(seq_loader.height * 0.70)
        counter = DirectionalLineCounter(
            line_start=(50, y_line),
            line_end=(seq_loader.width - 50, y_line),
            cooldown_frames=30
        )

    # 4. Optional visualizer
    video_writer = None
    visualizer = None
    if args.visualize:
        vis_out = ROOT / "outputs" / "visualizations" / f"{args.sequence}_{args.tracker}.mp4"
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        video_writer = VideoWriterWrapper(vis_out, fps=seq_loader.fps, frame_size=(seq_loader.width, seq_loader.height))
        visualizer = MOTVisualizer()

    # 5. Tracking execution loop
    all_predictions = []
    tracker.reset()

    for frame_idx in range(1, total_frames + 1):
        frame = seq_loader.get_frame(frame_idx)
        dets = dets_by_frame.get(frame_idx, [])

        tracks = tracker.update(dets, frame=frame if (args.use_reid and args.tracker == "hybrid_reid") else None)

        for t in tracks:
            x1, y1, x2, y2 = t.box
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            all_predictions.append((frame_idx, t.track_id, x1, y1, w, h, t.score))

        if counter is not None:
            counter.update(tracks, frame_id=frame_idx)

        if video_writer and frame is not None:
            vis_frame = visualizer.draw_frame(frame, tracks, frame_idx=frame_idx, fps=seq_loader.fps, counter=counter)
            video_writer.write(vis_frame)

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            logger.info(f"Processed frame {frame_idx}/{total_frames} (Active tracks: {len(tracks)})")

    if video_writer:
        video_writer.release()
        logger.info(f"Visualized output video saved to: {vis_out}")

    # 6. Save MOTChallenge prediction format
    pred_txt = ROOT / "outputs" / "predictions" / args.tracker / f"{args.sequence}.txt"
    write_mot_results(all_predictions, pred_txt)
    logger.info(f"Saved {len(all_predictions)} tracking predictions to {pred_txt}")

    # 7. Evaluate if ground truth exists
    gt_file = seq_path / "gt" / "gt.txt"
    if gt_file.exists():
        metrics = evaluate_sequence(gt_file, pred_txt, seq_name=args.sequence)
        logger.info("Evaluation Summary:")
        print(json.dumps(metrics, indent=2))

    if counter:
        counts = counter.get_counts()
        logger.info(f"Directional Counting: IN={counts['in']}, OUT={counts['out']}, TOTAL={counts['total']}")


if __name__ == "__main__":
    main()
