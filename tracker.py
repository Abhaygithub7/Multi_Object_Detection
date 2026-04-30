"""
tracker.py — Multi-object tracking module.

Wraps ByteTrack with configurable parameters for handling real-world
challenges: camera motion, occlusion, rapid movement, and scale changes.
"""

import supervision as sv
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TrackerConfig:
    """
    Configuration for ByteTrack tuned for sports footage.

    Attributes:
        track_activation_threshold:
            Detection confidence above which a track is immediately activated.
            Lower values allow weaker detections to seed new tracks (useful for
            partially-occluded or distant players).  Default lowered from 0.25
            to 0.20 for sports footage.

        lost_track_buffer:
            Number of frames a lost track is kept alive before being deleted.
            Higher values let the tracker "remember" a subject through short
            occlusions (e.g. a player briefly hidden behind another).
            Default raised from 30 to 45 (~1.5-1.8 s at 25-30 fps).

        minimum_matching_threshold:
            IoU threshold for matching a detection to an existing track.
            Lower values are more permissive, which helps when subjects
            move quickly between frames.  Default lowered from 0.8 to 0.7.

        frame_rate:
            Expected FPS of the source video.  Used internally by ByteTrack
            to scale its temporal heuristics.

        minimum_consecutive_frames:
            How many consecutive frames a detection must appear in before
            it is promoted to a confirmed track.  Raising this filters
            out spurious one-frame false positives.
    """
    track_activation_threshold: float = 0.20
    lost_track_buffer: int = 45
    minimum_matching_threshold: float = 0.70
    frame_rate: int = 25
    minimum_consecutive_frames: int = 3


class Tracker:
    """Multi-object tracker backed by ByteTrack."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
            minimum_consecutive_frames=self.config.minimum_consecutive_frames,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Feed new detections and return tracked detections with IDs."""
        return self._tracker.update_with_detections(detections)

    def reset(self):
        """Reset tracker state (useful between test cases)."""
        self._tracker.reset()

    # ── Convenience read-only properties for tests ──────────────────
    @property
    def activation_threshold(self) -> float:
        return self.config.track_activation_threshold

    @property
    def lost_track_buffer(self) -> int:
        return self.config.lost_track_buffer

    @property
    def matching_threshold(self) -> float:
        return self.config.minimum_matching_threshold

    @property
    def min_consecutive_frames(self) -> int:
        return self.config.minimum_consecutive_frames
