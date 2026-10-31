"""
Track Pool Manager and Lifecycle Finite State Machine Controller.
"""

from typing import List, Dict
import numpy as np

from src.tracking.track import Track, TrackState


class TrackManager:
    """
    Maintains track pools and coordinates lifecycle transitions:
    NEW -> TRACKED -> LOST -> EXPIRED
    """

    def __init__(self, track_buffer: int = 90, confirm_window: int = 2):
        self.track_buffer = track_buffer
        self.confirm_window = confirm_window

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.new_tracks: List[Track] = []
        self.expired_tracks: List[Track] = []

    def reset(self):
        Track.reset_counter()
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.new_tracks.clear()
        self.expired_tracks.clear()

    @property
    def all_active_tracks(self) -> List[Track]:
        """Return active candidates: confirmed tracked tracks + lost tracks within buffer."""
        return self.tracked_tracks + self.lost_tracks

    def predict_all(self):
        """Advance Kalman filters for all active and candidate tracks."""
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()
        for t in self.new_tracks:
            t.predict()

    def add_new_candidate(self, track: Track):
        """Add newly initiated track candidate."""
        self.new_tracks.append(track)

    def update_confirmed(self, track: Track):
        """Move track to confirmed tracked pool if confirmation criteria met."""
        if track in self.lost_tracks:
            self.lost_tracks.remove(track)
        if track in self.new_tracks:
            self.new_tracks.remove(track)

        if track not in self.tracked_tracks:
            self.tracked_tracks.append(track)

    def step_lifecycle(self):
        """
        Evaluate timeouts, promote confirmed tracks, and retire expired tracks.
        """
        # 1. Process candidate new tracks
        surviving_new: List[Track] = []
        for t in self.new_tracks:
            if t.hits >= self.confirm_window:
                t.state = TrackState.TRACKED
                self.tracked_tracks.append(t)
            elif t.missed_frames <= 1: # Give candidates 1 frame leeway
                surviving_new.append(t)
            else:
                t.mark_expired()
                self.expired_tracks.append(t)
        self.new_tracks = surviving_new

        # 2. Process tracked tracks (move missing tracks to LOST)
        surviving_tracked: List[Track] = []
        for t in self.tracked_tracks:
            if t.missed_frames > 0:
                t.mark_lost()
                self.lost_tracks.append(t)
            else:
                surviving_tracked.append(t)
        self.tracked_tracks = surviving_tracked

        # 3. Process lost tracks (retire tracks missing longer than track_buffer)
        surviving_lost: List[Track] = []
        for t in self.lost_tracks:
            if t.missed_frames > self.track_buffer:
                t.mark_expired()
                self.expired_tracks.append(t)
            else:
                surviving_lost.append(t)
        self.lost_tracks = surviving_lost

    def get_output_tracks(self) -> List[Track]:
        """Return only tracks that should be emitted in evaluation/output (confirmed active)."""
        return [t for t in self.tracked_tracks if t.is_confirmed and t.missed_frames == 0]

