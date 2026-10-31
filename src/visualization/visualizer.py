"""
Visualizer and Video Annotation Module for MOT17 Tracks and Counting.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.tracking.track import Track, TrackState
from src.counting.line_counter import DirectionalLineCounter


# Distinct color palette for track visualization
COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 56, 132),
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


class MOTVisualizer:
    """
    Renders bounding boxes, IDs, trajectories, track states, and directional counting lines.
    """

    def __init__(
        self,
        show_trails: bool = True,
        max_trail_len: int = 30,
        box_thickness: int = 2
    ):
        self.show_trails = show_trails
        self.max_trail_len = max_trail_len
        self.box_thickness = box_thickness

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_idx: int,
        fps: float = 30.0,
        counter: Optional[DirectionalLineCounter] = None
    ) -> np.ndarray:
        """
        Draw all tracking and counting annotations onto a copy of the frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw virtual counting line if present
        if counter is not None:
            pt1 = (int(counter.line_start[0]), int(counter.line_start[1]))
            pt2 = (int(counter.line_end[0]), int(counter.line_end[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 255), 3)
            # Draw line normal / direction indicator
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            arrow_end = (int(mid[0] + counter.line_normal[0] * 35), int(mid[1] + counter.line_normal[1] * 35))
            cv2.arrowedLine(vis, mid, arrow_end, (0, 200, 255), 2, tipLength=0.3)
            cv2.putText(vis, "COUNT LINE (IN ->)", (pt1[0] + 10, pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. Draw tracks
        for track in tracks:
            color = get_color(track.track_id)
            x1, y1, x2, y2 = map(int, track.box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw trajectory trail
            if self.show_trails and len(track.trajectory) > 1:
                pts = track.trajectory[-self.max_trail_len:]
                for i in range(1, len(pts)):
                    p_prev = (int(pts[i - 1][1]), int(pts[i - 1][2]))
                    p_curr = (int(pts[i][1]), int(pts[i][2]))
                    thickness = int(round(1 + (i / len(pts)) * 2))
                    cv2.line(vis, p_prev, p_curr, color, thickness)

            # Draw ID label tag
            label = f"ID: {track.track_id} ({track.score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw statistics overlay dashboard
        overlay_h, overlay_w = 95, 340
        cv2.rectangle(vis, (15, 15), (15 + overlay_w, 15 + overlay_h), (20, 20, 20), -1)
        cv2.rectangle(vis, (15, 15), (15 + overlay_w, 15 + overlay_h), (80, 80, 80), 1)

        cv2.putText(vis, f"Frame: {frame_idx:04d} | FPS: {fps:.1f}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        active_count = len([t for t in tracks if t.missed_frames == 0])
        cv2.putText(vis, f"Active Tracks: {active_count}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if counter is not None:
            counts = counter.get_counts()
            cv2.putText(vis, f"IN: {counts['in']} | OUT: {counts['out']} | Total: {counts['total']}", (25, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return vis


class VideoWriterWrapper:
    """Wrapper around OpenCV VideoWriter."""

    def __init__(self, output_path: Path, fps: float, frame_size: Tuple[int, int]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, frame_size)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

