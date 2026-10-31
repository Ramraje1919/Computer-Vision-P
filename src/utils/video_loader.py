"""
Frame and sequence loading utilities for MOT17 sequences.
"""

from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional
import configparser
import cv2
import numpy as np


class MOTSequenceLoader:
    """Loads image frames and sequence metadata from an MOT17 sequence directory."""

    def __init__(self, sequence_path: Path):
        self.sequence_path = Path(sequence_path)
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence directory not found: {self.sequence_path}")

        self.seqinfo = self._load_seqinfo()
        self.img_dir = self.sequence_path / self.seqinfo.get("imDir", "img1")
        self.ext = self.seqinfo.get("imExt", ".jpg")
        self.total_frames = self.seqinfo.get("seqLength", 0)
        self.fps = self.seqinfo.get("frameRate", 30.0)
        self.width = self.seqinfo.get("imWidth", 1920)
        self.height = self.seqinfo.get("imHeight", 1080)

    def _load_seqinfo(self) -> Dict[str, Any]:
        info_file = self.sequence_path / "seqinfo.ini"
        if not info_file.exists():
            return {"imDir": "img1", "imExt": ".jpg", "frameRate": 30.0, "seqLength": 0, "imWidth": 1920, "imHeight": 1080}
        
        config = configparser.ConfigParser()
        config.read(info_file)
        sec = config["Sequence"]
        return {
            "name": sec.get("name", self.sequence_path.name),
            "imDir": sec.get("imDir", "img1"),
            "frameRate": float(sec.get("frameRate", 30)),
            "seqLength": int(sec.get("seqLength", 0)),
            "imWidth": int(sec.get("imWidth", 1920)),
            "imHeight": int(sec.get("imHeight", 1080)),
            "imExt": sec.get("imExt", ".jpg"),
        }

    def __len__(self) -> int:
        return self.total_frames

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Fetch frame by 1-based index."""
        frame_file = self.img_dir / f"{frame_idx:06d}{self.ext}"
        if not frame_file.exists():
            return None
        return cv2.imread(str(frame_file))

    def iter_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_number, frame_bgr) tuples for all frames in sequence."""
        for idx in range(1, self.total_frames + 1):
            frame = self.get_frame(idx)
            if frame is not None:
                yield idx, frame

