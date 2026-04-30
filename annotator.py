"""
annotator.py — Frame annotation module.

Wraps supervision annotators into a single FrameAnnotator class that
applies bounding boxes, labels, trajectory traces, and an object-count
overlay.  Each feature can be toggled independently via AnnotatorConfig.
Supports team-specific colour coding via per-detection custom colors.
"""

import cv2
import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


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
    box_thickness: int = 2
    label_text_scale: float = 0.5


class FrameAnnotator:
    """Applies all visual annotations to a single frame."""

    def __init__(self, config: Optional[AnnotatorConfig] = None):
        self.config = config or AnnotatorConfig()

        # We use ColorLookup.TRACK because our detections carry tracker_id
        # but not always class_id.
        if self.config.show_boxes:
            self._box = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.TRACK,
                thickness=self.config.box_thickness,
            )
        if self.config.show_labels:
            self._label = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.TRACK,
                text_scale=self.config.label_text_scale,
            )
        if self.config.show_traces:
            self._trace = sv.TraceAnnotator(
                trace_length=self.config.trace_length,
                color_lookup=sv.ColorLookup.TRACK,
            )

        # Optional: per-detection custom colors set by set_custom_colors()
        self._custom_color_palette: Optional[sv.ColorPalette] = None

    # ── Public API ──────────────────────────────────────────────────

    def set_team_colors(self, team_colors_bgr: Dict[int, Tuple[int, int, int]]):
        """
        Set custom per-team box colours.  *team_colors_bgr* maps
        team_label → (B, G, R).  These are stored and applied
        during annotate() via custom_labels parameter.
        """
        self._team_colors_bgr = team_colors_bgr

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        custom_labels: Optional[List[str]] = None,
        team_labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Annotate *frame* with the given *detections* and return the result.

        Args:
            frame:          Source video frame (BGR).
            detections:     Tracked detections with tracker_id.
            custom_labels:  Optional list of label strings (e.g. "Messi #10 [RMA]").
                            If None, falls back to build_labels().
            team_labels:    Optional list of team ints per detection.
                            Used to draw team-coloured boxes manually.

        The original frame is never mutated; a copy is returned.
        """
        out = frame.copy()

        # 1. Trajectory traces (drawn first, behind boxes)
        if (
            self.config.show_traces
            and detections.tracker_id is not None
            and len(detections) > 0
        ):
            out = self._trace.annotate(scene=out, detections=detections)

        # Determine if we should use per-detection team colours
        use_team_colors = (
            team_labels is not None
            and hasattr(self, "_team_colors_bgr")
            and self._team_colors_bgr
        )

        if use_team_colors and len(detections) > 0:
            # Draw team-coloured boxes and labels manually per detection
            labels = custom_labels or self.build_labels(detections)
            for i in range(len(detections)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                tl = team_labels[i]
                bgr = self._team_colors_bgr.get(tl, (255, 255, 255))

                # Box
                if self.config.show_boxes:
                    cv2.rectangle(out, (x1, y1), (x2, y2), bgr, self.config.box_thickness)

                # Label background + text
                if self.config.show_labels and i < len(labels):
                    label = labels[i]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = self.config.label_text_scale
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                    # Label background
                    cv2.rectangle(
                        out,
                        (x1, y1 - th - baseline - 4),
                        (x1 + tw + 4, y1),
                        bgr,
                        cv2.FILLED,
                    )
                    # Label text (black on coloured background)
                    cv2.putText(
                        out, label,
                        (x1 + 2, y1 - baseline - 2),
                        font, scale, (0, 0, 0), thickness, cv2.LINE_AA,
                    )
        else:
            # Fallback: use supervision annotators with default colours
            if self.config.show_boxes:
                out = self._box.annotate(scene=out, detections=detections)

            if self.config.show_labels and len(detections) > 0:
                labels = custom_labels or self.build_labels(detections)
                out = self._label.annotate(
                    scene=out, detections=detections, labels=labels,
                )

        # Object-count overlay
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
