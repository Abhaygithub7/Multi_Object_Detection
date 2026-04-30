"""
annotator.py — Frame annotation module.

Wraps supervision annotators into a single FrameAnnotator class that
applies bounding boxes, labels, trajectory traces, and an object-count
overlay.  Each feature can be toggled independently via AnnotatorConfig.
"""

import cv2
import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class AnnotatorConfig:
    """
    Configuration for all visual annotation features.

    Attributes:
        show_boxes:        Draw bounding boxes around tracked subjects.
        show_labels:       Show '#ID confidence' labels above each box.
        show_traces:       Draw a fading trajectory trail behind each subject.
        show_object_count: Render a live 'Objects Tracked: N' counter.
        trace_length:      Number of past frames used for the trace trail.
        count_position:    (x, y) pixel position for the count text.
        count_color:       BGR colour tuple for the count text.
        count_font_scale:  Font size for the count overlay.
        count_thickness:   Stroke thickness for the count text.
    """
    show_boxes: bool = True
    show_labels: bool = True
    show_traces: bool = True
    show_object_count: bool = True
    trace_length: int = 60
    count_position: Tuple[int, int] = (20, 50)
    count_color: Tuple[int, int, int] = (0, 255, 0)
    count_font_scale: float = 1.0
    count_thickness: int = 2


class FrameAnnotator:
    """Applies all visual annotations to a single frame."""

    def __init__(self, config: Optional[AnnotatorConfig] = None):
        self.config = config or AnnotatorConfig()

        # Only instantiate the annotators we actually need.
        # We use ColorLookup.TRACK because our detections carry tracker_id
        # but not class_id.
        if self.config.show_boxes:
            self._box = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.TRACK,
            )
        if self.config.show_labels:
            self._label = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.TRACK,
            )
        if self.config.show_traces:
            self._trace = sv.TraceAnnotator(
                trace_length=self.config.trace_length,
                color_lookup=sv.ColorLookup.TRACK,
            )

    # ── Public API ──────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotate *frame* with the given *detections* and return the result.

        The original frame is never mutated; a copy is returned.
        """
        out = frame.copy()

        # 1. Trajectory traces (drawn first, behind boxes)
        #    TraceAnnotator requires tracker_id; skip if absent or empty.
        if (
            self.config.show_traces
            and detections.tracker_id is not None
            and len(detections) > 0
        ):
            out = self._trace.annotate(scene=out, detections=detections)

        # 2. Bounding boxes
        if self.config.show_boxes:
            out = self._box.annotate(scene=out, detections=detections)

        # 3. Labels
        if self.config.show_labels and len(detections) > 0:
            labels = self.build_labels(detections)
            out = self._label.annotate(
                scene=out, detections=detections, labels=labels,
            )

        # 4. Object-count overlay
        if self.config.show_object_count:
            text = self.build_count_text(detections)
            cv2.putText(
                out,
                text,
                self.config.count_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.count_font_scale,
                self.config.count_color,
                self.config.count_thickness,
                cv2.LINE_AA,
            )

        return out

    # ── Helper builders (public so tests can inspect them) ─────────

    def build_labels(self, detections: sv.Detections) -> list[str]:
        """Build the '#ID confidence' label list for the given detections."""
        if detections.tracker_id is None:
            return [f"{c:0.2f}" for c in detections.confidence]
        return [
            f"#{tid} {conf:0.2f}"
            for conf, tid in zip(detections.confidence, detections.tracker_id)
        ]

    def build_count_text(self, detections: sv.Detections) -> str:
        """Build the object-count string."""
        return f"Objects Tracked: {len(detections)}"
