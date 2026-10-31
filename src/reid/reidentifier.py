"""
Re-Identification and Candidate Matching Utilities.
"""

from typing import List, Optional
import numpy as np

from src.reid.feature_gallery import FeatureGallery


def compute_appearance_distance_matrix(
    galleries: List[FeatureGallery],
    candidate_features: List[Optional[np.ndarray]],
    max_distance: float = 1.0
) -> np.ndarray:
    """
    Construct appearance distance matrix between track galleries and candidate features.
    Shape: (len(galleries), len(candidate_features))
    """
    num_tracks = len(galleries)
    num_candidates = len(candidate_features)

    if num_tracks == 0 or num_candidates == 0:
        return np.empty((num_tracks, num_candidates), dtype=np.float32)

    cost_matrix = np.full((num_tracks, num_candidates), max_distance, dtype=np.float32)

    for i, gallery in enumerate(galleries):
        for j, feat in enumerate(candidate_features):
            if feat is not None and not gallery.is_empty():
                cost_matrix[i, j] = gallery.distance_to(feat)

    return cost_matrix

