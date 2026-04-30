"""
tests/test_annotator.py — TDD tests for Phase 2 Features 2, 3, 5:
    Feature 2: Basic visual annotations (bounding boxes, IDs)
    Feature 3: Trajectory visualization (fading trails)
    Feature 5: Object count overlay

These tests define the contract for the FrameAnnotator module.
"""

import pytest
import numpy as np
import supervision as sv


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _blank_frame(h=480, w=640):
    """Create a blank black frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detections(boxes, confidences=None, tracker_ids=None):
    """Build a supervision Detections object with tracker IDs attached."""
    if len(boxes) == 0:
        return sv.Detections.empty()

    xyxy = np.array(boxes, dtype=np.float32)
    if confidences is None:
        confidences = np.ones(len(boxes), dtype=np.float32) * 0.9
    else:
        confidences = np.array(confidences, dtype=np.float32)

    dets = sv.Detections(xyxy=xyxy, confidence=confidences)

    if tracker_ids is not None:
        dets.tracker_id = np.array(tracker_ids, dtype=int)

    return dets


# ═══════════════════════════════════════════════════════════════════════════
#  1. AnnotatorConfig Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAnnotatorConfig:
    """Verify the AnnotatorConfig exposes sensible defaults."""

    def test_default_config_has_all_features_enabled(self):
        """By default, boxes, labels, traces, and object count are all ON."""
        from annotator import AnnotatorConfig
        cfg = AnnotatorConfig()
        assert cfg.show_boxes is True
        assert cfg.show_labels is True
        assert cfg.show_traces is True
        assert cfg.show_object_count is True

    def test_trace_length_default(self):
        """Trace length should default to 60 frames (~2s at 30fps)."""
        from annotator import AnnotatorConfig
        cfg = AnnotatorConfig()
        assert cfg.trace_length == 60

    def test_count_position_default(self):
        """Object count text should be drawn at (20, 50) by default."""
        from annotator import AnnotatorConfig
        cfg = AnnotatorConfig()
        assert cfg.count_position == (20, 50)

    def test_count_color_default(self):
        """Object count colour should be green (0, 255, 0) by default."""
        from annotator import AnnotatorConfig
        cfg = AnnotatorConfig()
        assert cfg.count_color == (0, 255, 0)

    def test_features_can_be_toggled_off(self):
        """All visual features can be individually disabled."""
        from annotator import AnnotatorConfig
        cfg = AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=False,
            show_object_count=False,
        )
        assert cfg.show_boxes is False
        assert cfg.show_labels is False
        assert cfg.show_traces is False
        assert cfg.show_object_count is False


# ═══════════════════════════════════════════════════════════════════════════
#  2. Annotation Output Tests — Feature 2 (bounding boxes + IDs)
# ═══════════════════════════════════════════════════════════════════════════

class TestBoxAndLabelAnnotation:
    """Feature 2: annotated frame must contain bounding boxes and ID labels."""

    def test_annotate_returns_same_shape(self):
        """The annotated frame must preserve the original dimensions."""
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        frame = _blank_frame(480, 640)
        dets = _make_detections(
            [[50, 50, 150, 150]], tracker_ids=[1]
        )
        result = ann.annotate(frame, dets)
        assert result.shape == frame.shape

    def test_annotate_modifies_pixels_when_detections_present(self):
        """When there are detections, the frame must be visually altered."""
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        frame = _blank_frame(480, 640)
        dets = _make_detections(
            [[50, 50, 150, 150]], tracker_ids=[1]
        )
        result = ann.annotate(frame, dets)
        # At least some pixels should have changed from the blank black frame
        assert not np.array_equal(result, _blank_frame(480, 640)), (
            "Annotated frame should differ from a blank frame when detections exist"
        )

    def test_annotate_no_crash_on_empty_detections(self):
        """Passing empty detections must not crash."""
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        frame = _blank_frame()
        dets = _make_detections([])
        result = ann.annotate(frame, dets)
        assert result.shape == frame.shape

    def test_boxes_disabled_leaves_frame_unchanged_near_detection(self):
        """
        With show_boxes=False and show_labels=False and show_traces=False
        and show_object_count=False, the output should be identical to the input.
        """
        from annotator import FrameAnnotator, AnnotatorConfig
        cfg = AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=False,
            show_object_count=False,
        )
        ann = FrameAnnotator(cfg)
        frame = _blank_frame()
        dets = _make_detections(
            [[50, 50, 150, 150]], tracker_ids=[1]
        )
        result = ann.annotate(frame, dets)
        assert np.array_equal(result, frame), (
            "With all features disabled, the output should be a copy of the input"
        )

    def test_label_format_includes_id_and_confidence(self):
        """Generated labels must include '#ID confidence' format."""
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        dets = _make_detections(
            [[50, 50, 150, 150]], confidences=[0.85], tracker_ids=[7]
        )
        labels = ann.build_labels(dets)
        assert len(labels) == 1
        assert "#7" in labels[0]
        assert "0.85" in labels[0]


# ═══════════════════════════════════════════════════════════════════════════
#  3. Trajectory Visualization Tests — Feature 3
# ═══════════════════════════════════════════════════════════════════════════

class TestTrajectoryVisualization:
    """Feature 3: Traces must be drawn when enabled."""

    def test_traces_drawn_over_multiple_frames(self):
        """
        After feeding the same detection across several frames,
        traces should be rendered (pixels changed in trace region).
        """
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=True,
            show_object_count=False,
        ))

        # Simulate a detection moving right over 10 frames
        for i in range(10):
            x = 100 + i * 10
            frame = _blank_frame()
            dets = _make_detections(
                [[x, 200, x + 50, 250]], tracker_ids=[1]
            )
            result = ann.annotate(frame, dets)

        # By frame 10, there should be trace pixels drawn
        # (the trace is behind the box, on the black canvas)
        assert not np.array_equal(result, _blank_frame()), (
            "Traces should alter frame pixels after multiple frames of movement"
        )

    def test_traces_disabled_no_trail(self):
        """With show_traces=False, no trace pixels should appear."""
        from annotator import FrameAnnotator, AnnotatorConfig
        cfg = AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=False,
            show_object_count=False,
        )
        ann = FrameAnnotator(cfg)

        for i in range(10):
            x = 100 + i * 10
            frame = _blank_frame()
            dets = _make_detections(
                [[x, 200, x + 50, 250]], tracker_ids=[1]
            )
            result = ann.annotate(frame, dets)

        # With all features off, result should equal a blank frame
        assert np.array_equal(result, _blank_frame()), (
            "No traces should be drawn when show_traces is False"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Object Count Overlay Tests — Feature 5
# ═══════════════════════════════════════════════════════════════════════════

class TestObjectCountOverlay:
    """Feature 5: A live object count must be rendered on the frame."""

    def test_object_count_changes_pixels_in_text_region(self):
        """When object count is enabled, text must be drawn on the frame."""
        from annotator import FrameAnnotator, AnnotatorConfig
        cfg = AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=False,
            show_object_count=True,
        )
        ann = FrameAnnotator(cfg)
        frame = _blank_frame()
        dets = _make_detections(
            [[50, 50, 150, 150], [200, 200, 300, 300]],
            tracker_ids=[1, 2],
        )
        result = ann.annotate(frame, dets)

        # Check that pixels in the text region (top-left corner) were modified
        text_region = result[20:70, 5:400]
        assert np.any(text_region > 0), (
            "Object count text should draw pixels in the top-left region"
        )

    def test_object_count_shows_correct_number(self):
        """
        The count text builder must return the correct count for
        the given number of detections.
        """
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        dets = _make_detections(
            [[50, 50, 150, 150], [200, 200, 300, 300], [400, 400, 500, 500]],
            tracker_ids=[1, 2, 3],
        )
        text = ann.build_count_text(dets)
        assert "3" in text

    def test_object_count_zero_when_no_detections(self):
        """When there are no detections, count should show 0."""
        from annotator import FrameAnnotator, AnnotatorConfig
        ann = FrameAnnotator(AnnotatorConfig())
        dets = _make_detections([])
        text = ann.build_count_text(dets)
        assert "0" in text

    def test_object_count_disabled(self):
        """With show_object_count=False, no text pixels in the count region."""
        from annotator import FrameAnnotator, AnnotatorConfig
        cfg = AnnotatorConfig(
            show_boxes=False,
            show_labels=False,
            show_traces=False,
            show_object_count=False,
        )
        ann = FrameAnnotator(cfg)
        frame = _blank_frame()
        dets = _make_detections(
            [[50, 50, 150, 150]], tracker_ids=[1]
        )
        result = ann.annotate(frame, dets)
        text_region = result[20:70, 5:400]
        assert not np.any(text_region > 0), (
            "No text should appear when show_object_count is False"
        )
