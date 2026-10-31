"""
Batch Person Re-ID Feature Extractor using OSNet.
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.reid.osnet import build_osnet_model


class OSNetFeatureExtractor:
    """Extracts 512-d L2-normalized ReID features from cropped bounding boxes."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
        crop_size: tuple = (256, 128) # (height, width)
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.crop_h, self.crop_w = crop_size
        self.model = build_osnet_model(checkpoint_path, device=self.device)

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def extract_crop(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single cropped and normalized tensor patch."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        # Transpose to (C, H, W)
        return crop.transpose(2, 0, 1)

    @torch.no_grad()
    def extract_batch(self, frame: np.ndarray, boxes: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract normalized 512-d embeddings for a list of [x1, y1, x2, y2] boxes."""
        if len(boxes) == 0:
            return []

        patches = []
        valid_indices = []
        for idx, box in enumerate(boxes):
            patch = self.extract_crop(frame, box)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(idx)

        results: List[Optional[np.ndarray]] = [None] * len(boxes)
        if not patches:
            return results

        batch_tensor = torch.from_numpy(np.stack(patches, axis=0)).to(self.device)
        features = self.model(batch_tensor)
        features = F.normalize(features, p=2, dim=1)
        feat_np = features.cpu().numpy()

        for i, orig_idx in enumerate(valid_indices):
            results[orig_idx] = feat_np[i]

        return results

