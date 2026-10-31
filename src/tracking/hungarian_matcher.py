"""
Hungarian / Munkres Linear Sum Assignment Wrapper.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve bipartite matching on cost matrix subject to a maximum cost threshold.

    Returns:
        matches: List of (row_idx, col_idx) pairs where cost <= threshold
        unmatched_rows: List of row indices not matched or whose cost > threshold
        unmatched_cols: List of column indices not matched or whose cost > threshold
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_rows: List[int] = list(range(cost_matrix.shape[0]))
    unmatched_cols: List[int] = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            if r in unmatched_rows:
                unmatched_rows.remove(r)
            if c in unmatched_cols:
                unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols

