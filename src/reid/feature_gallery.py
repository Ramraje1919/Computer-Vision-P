"""
Multi-Sample Exemplar Feature Gallery with Occlusion Quality Gating.
"""

from typing import List, Optional
import numpy as np


class FeatureGallery:
    """
    Maintains a multi-sample appearance gallery per track.
    Features are protected against corruption during occlusion by strict quality gating.
    """

    def __init__(
        self,
        max_samples: int = 6,
        ema_momentum: float = 0.90,
        min_update_conf: float = 0.60,
        max_update_overlap: float = 0.30
    ):
        self.max_samples = max_samples
        self.ema_momentum = ema_momentum
        self.min_update_conf = min_update_conf
        self.max_update_overlap = max_update_overlap

        self.ema_feature: Optional[np.ndarray] = None
        self.samples: List[np.ndarray] = []

    def is_empty(self) -> bool:
        return self.ema_feature is None and len(self.samples) == 0

    def add_initial_feature(self, feature: np.ndarray):
        """Initialize the gallery with the first observed appearance signature."""
        feat = feature / np.linalg.norm(feature)
        self.ema_feature = feat.copy()
        self.samples.append(feat.copy())

    def update(
        self,
        feature: Optional[np.ndarray],
        score: float = 1.0,
        max_overlap_with_others: float = 0.0
    ):
        """
        Conditionally update the gallery only when appearance quality is high.
        Rejects updating during heavy occlusion or high ambiguity.
        """
        if feature is None:
            return

        feat = feature / np.linalg.norm(feature)

        # Quality check: reject updating if confidence is low or crop overlaps another pedestrian
        if score < self.min_update_conf or max_overlap_with_others > self.max_update_overlap:
            return

        if self.ema_feature is None:
            self.add_initial_feature(feat)
            return

        # Update EMA running representation
        self.ema_feature = self.ema_momentum * self.ema_feature + (1.0 - self.ema_momentum) * feat
        self.ema_feature /= np.linalg.norm(self.ema_feature)

        # Manage diverse exemplar pool
        # Only add a new sample if it has distinct diversity (cosine distance > 0.15 from existing)
        distances = [1.0 - np.dot(feat, s) for s in self.samples]
        if min(distances) > 0.15:
            if len(self.samples) >= self.max_samples:
                # Evict oldest sample, preserving the very first exemplar (canonical appearance)
                self.samples.pop(1 if len(self.samples) > 1 else 0)
            self.samples.append(feat.copy())

    def distance_to(self, feature: Optional[np.ndarray]) -> float:
        """
        Compute minimum cosine distance between candidate feature and all gallery exemplars.
        Returns: Cosine distance in range [0.0, 2.0] (0 = identical, 1 = orthogonal).
        """
        if feature is None or self.is_empty():
            return 1.0

        feat = feature / np.linalg.norm(feature)

        # Compare against EMA representation
        dist_ema = 1.0 - float(np.clip(np.dot(self.ema_feature, feat), -1.0, 1.0))

        # Compare against all historical exemplars
        sample_dists = [1.0 - float(np.clip(np.dot(s, feat), -1.0, 1.0)) for s in self.samples]

        # Return minimum distance across gallery
        return min(dist_ema, min(sample_dists))

