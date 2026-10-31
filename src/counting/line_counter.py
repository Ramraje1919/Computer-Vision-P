"""
Directional Virtual Line Person Counter with Anti-Jitter Cooldown.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.tracking.track import Track


def line_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """
    Check if line segment p1->p2 intersects line segment p3->p4.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


class DirectionalLineCounter:
    """
    Directional person counter tracking line crossings (IN / OUT).
    Operates strictly on confirmed track trajectory histories.
    """

    def __init__(
        self,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        cooldown_frames: int = 30,
        trajectory_smoothing_window: int = 5
    ):
        self.line_start = np.array(line_start, dtype=float)
        self.line_end = np.array(line_end, dtype=float)
        self.cooldown_frames = cooldown_frames
        self.smoothing_window = trajectory_smoothing_window

        # Line direction vector and normal vector (pointing right/inward)
        line_vec = self.line_end - self.line_start
        self.line_normal = np.array([line_vec[1], -line_vec[0]], dtype=float)
        norm_len = np.linalg.norm(self.line_normal)
        if norm_len > 1e-6:
            self.line_normal /= norm_len

        self.in_count = 0
        self.out_count = 0
        self.crossing_events: List[Dict[str, Any]] = []

        # Track ID -> last crossing frame number (cooldown tracker)
        self._last_crossing_frame: Dict[int, int] = {}
        # Track ID -> previous side of the line (+1 or -1)
        self._track_side: Dict[int, int] = {}

    def reset(self):
        self.in_count = 0
        self.out_count = 0
        self.crossing_events.clear()
        self._last_crossing_frame.clear()
        self._track_side.clear()

    def _get_side(self, point: Tuple[float, float]) -> int:
        """Determine which side of the directed line a point lies on (+1 or -1)."""
        dx = self.line_end[0] - self.line_start[0]
        dy = self.line_end[1] - self.line_start[1]
        vx = point[0] - self.line_start[0]
        vy = point[1] - self.line_start[1]
        cross_prod = dx * vy - dy * vx
        return 1 if cross_prod > 0 else -1

    def update(self, tracks: List[Track], frame_id: int):
        """
        Evaluate line crossing for all active confirmed tracks.
        """
        for track in tracks:
            if not track.is_confirmed:
                continue

            track_id = track.track_id
            traj = track.trajectory
            if len(traj) == 0:
                continue

            curr_pt = (traj[-1][1], traj[-1][2])
            curr_side = self._get_side(curr_pt)

            if len(traj) < 2:
                self._track_side[track_id] = curr_side
                continue

            prev_pt = (traj[-2][1], traj[-2][2])
            prev_side = self._track_side.get(track_id, self._get_side(prev_pt))

            # Check if crossing occurred
            if curr_side != prev_side:
                # Check cooldown to prevent duplicate counting from jitter near boundary
                last_frame = self._last_crossing_frame.get(track_id, -999)
                if frame_id - last_frame > self.cooldown_frames:
                    # Verify geometric segment intersection
                    intersects = line_segment_intersection(
                        prev_pt,
                        curr_pt,
                        (self.line_start[0], self.line_start[1]),
                        (self.line_end[0], self.line_end[1])
                    )

                    if intersects:
                        # Determine direction via motion vector dot product with line normal
                        motion_vec = np.array(curr_pt) - np.array(prev_pt)
                        dot_prod = np.dot(motion_vec, self.line_normal)

                        direction = "IN" if dot_prod > 0 else "OUT"
                        if direction == "IN":
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        self._last_crossing_frame[track_id] = frame_id
                        self.crossing_events.append({
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "direction": direction,
                            "in_total": self.in_count,
                            "out_total": self.out_count,
                            "crossing_pos": curr_pt
                        })

                self._track_side[track_id] = curr_side

    def get_counts(self) -> Dict[str, int]:
        return {
            "in": self.in_count,
            "out": self.out_count,
            "total": self.in_count + self.out_count
        }
