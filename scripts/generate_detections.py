"""
Precompute and cache detections for MOT17 sequences to accelerate tracker experiments.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.video_loader import MOTSequenceLoader
from src.detection.detector import YOLODetector, PublicDetectionLoader, DetectionCache, Detection

logger = get_logger("GenerateDetections")


def cache_sequence_detections(
    sequence_path: Path,
    output_cache_path: Path,
    detector_type: str = "public",
    yolo_weights: str = "yolo11n.pt",
    conf_thresh: float = 0.05,
    img_size: int = 960
):
    sequence_path = Path(sequence_path)
    output_cache_path = Path(output_cache_path)

    if output_cache_path.exists():
        logger.info(f"Cache already exists at: {output_cache_path}. Skipping.")
        return

    logger.info(f"Processing sequence: {sequence_path.name} (Detector: {detector_type})")

    if detector_type == "public":
        det_file = sequence_path / "det" / "det.txt"
        if not det_file.exists():
            raise FileNotFoundError(f"det.txt not found in {sequence_path}")
        loader = PublicDetectionLoader(det_file)
        detections = loader.detections_by_frame
    elif detector_type == "yolo":
        yolo = YOLODetector(weights_path=yolo_weights, conf_threshold=conf_thresh, img_size=img_size)
        seq_loader = MOTSequenceLoader(sequence_path)
        detections: Dict[int, List[Detection]] = {}

        total = len(seq_loader)
        for frame_idx, frame in seq_loader.iter_frames():
            dets = yolo.predict(frame)
            detections[frame_idx] = dets
            if frame_idx % 100 == 0 or frame_idx == total:
                logger.info(f"Frame {frame_idx}/{total} processed ({len(dets)} detections)")
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    DetectionCache.save(detections, output_cache_path)
    total_dets = sum(len(d) for d in detections.values())
    logger.info(f"Successfully cached {total_dets} detections for {sequence_path.name} to {output_cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Cache detections for MOT17 sequences")
    parser.add_argument("--sequence", type=str, default="all_train", help="Sequence name or 'all_train' / 'validation'")
    parser.add_argument("--detector", type=str, choices=["public", "yolo"], default="public")
    parser.add_argument("--split-config", type=str, default="config/dataset_split.yaml")
    args = parser.parse_args()

    cache_dir = ROOT / "outputs" / "detections_cache" / args.detector
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.split_config, "r", encoding="utf-8") as f:
        split_data = yaml.safe_load(f)

    train_root = ROOT / "data" / "MOT17" / "train"

    seq_targets = []
    if args.sequence == "all_train":
        seq_targets = [d.name for d in train_root.iterdir() if d.is_dir()]
    elif args.sequence == "validation":
        val_bases = [s["name"] for s in split_data["splits"]["validation"]["unique_sequences"]]
        seq_targets = [d.name for d in train_root.iterdir() if d.is_dir() if any(d.name.startswith(b) for b in val_bases)]
    else:
        seq_targets = [args.sequence]

    for seq_name in seq_targets:
        seq_dir = train_root / seq_name
        cache_path = cache_dir / f"{seq_name}.json"
        cache_sequence_detections(seq_dir, cache_path, detector_type=args.detector)


if __name__ == "__main__":
    main()

