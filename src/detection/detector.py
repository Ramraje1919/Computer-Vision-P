"""
Unified Detection Interface and Caching System for MOT17.

Supports:
1. YOLO detection (YOLOv8 / YOLO11) via Ultralytics
2. Official MOT17 public detections (DPM, FRCNN, SDP)
3. Offline detection caching to accelerate tracking experiments
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.detection.postprocessing import nms, clip_boxes, xywh_to_xyxy


class Detection:
    """Represents a single pedestrian detection box."""

    def __init__(self, box: np.ndarray, score: float, class_id: int = 0):
        # box is [x1, y1, x2, y2]
        self.box = np.asarray(box, dtype=float)
        self.score = float(score)
        self.class_id = int(class_id)

    @property
    def xyxy(self) -> np.ndarray:
        return self.box

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.tolist(),
            "score": round(self.score, 4),
            "class_id": self.class_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(box=np.array(data["box"], dtype=float), score=data["score"], class_id=data.get("class_id", 0))


class PublicDetectionLoader:
    """Loads detections from MOT17 det/det.txt file."""

    def __init__(self, det_txt_path: Path):
        self.det_txt_path = Path(det_txt_path)
        if not self.det_txt_path.exists():
            raise FileNotFoundError(f"det.txt not found: {self.det_txt_path}")
        self.detections_by_frame = self._parse()

    def _parse(self) -> Dict[int, List[Detection]]:
        dets: Dict[int, List[Detection]] = {}
        with open(self.det_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    frame_idx = int(float(parts[0]))
                    # Format: frame, -1, x, y, w, h, conf
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                    box_xyxy = xywh_to_xyxy(np.array([x, y, w, h]))
                    det = Detection(box=box_xyxy, score=conf, class_id=0)
                    dets.setdefault(frame_idx, []).append(det)
        return dets

    def get_detections(self, frame_idx: int) -> List[Detection]:
        return self.detections_by_frame.get(frame_idx, [])


class YOLODetector:
    """YOLO Person Detector wrapper."""

    def __init__(
        self,
        weights_path: str = "yolo11n.pt",
        conf_threshold: float = 0.10,
        img_size: int = 960,
        device: Optional[str] = None
    ):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed.")
        self.weights_path = weights_path
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame and return list of Detection objects."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            classes=[0], # Class 0 = Person in COCO
            conf=self.conf_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        xyxy = clip_boxes(xyxy, width=w, height=h)

        return [Detection(box=xyxy[i], score=scores[i], class_id=0) for i in range(len(scores))]


class DetectionCache:
    """Saves and loads precomputed detections to/from disk to eliminate redundant detector runs."""

    @staticmethod
    def save(detections: Dict[int, List[Detection]], cache_path: Path):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            str(frame): [d.to_dict() for d in det_list]
            for frame, det_list in detections.items()
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    @staticmethod
    def load(cache_path: Path) -> Dict[int, List[Detection]]:
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file does not exist: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            int(frame): [Detection.from_dict(d) for d in det_list]
            for frame, det_list in raw.items()
        }

