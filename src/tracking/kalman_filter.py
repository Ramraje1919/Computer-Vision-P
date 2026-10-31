"""
BoT-SORT-Style 8-State Kalman Filter for Bounding Box Motion Estimation.

State vector:
    x = [cx, cy, w, h, vx, vy, vw, vh]^T

Directly estimates width (w) and height (h) rather than aspect ratio (w/h),
eliminating the numerical instability and aspect-ratio drift seen in classical SORT.
"""

from typing import Tuple
import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    Linear Kalman Filter operating on 2D bounding boxes in (cx, cy, w, h) coordinates.
    """

    def __init__(self, std_weight_position: float = 0.05, std_weight_velocity: float = 0.00625):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track state from unassociated measurement [cx, cy, w, h].
        Returns: (mean, covariance)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state mean and covariance forward by one time step.
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space [cx, cy, w, h].
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct state mean and covariance given new observation [cx, cy, w, h].
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

