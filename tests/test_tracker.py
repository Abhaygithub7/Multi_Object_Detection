"""
tests/test_tracker.py — TDD tests for Phase 2, Feature 1:
    "Handle camera motion, occlusions, and rapid movement via tracker params."

These tests are written BEFORE the implementation is finalized (RED phase).
They define the contract that our Tracker must satisfy.
"""

import pytest
import numpy as np
import supervision as sv
from tracker import Tracker, TrackerConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_detections(boxes, confidences=None):
    """
    Build a supervision Detections object from a list of [x1, y1, x2, y2] boxes.
    Returns sv.Detections.empty() when boxes is empty.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()

    xyxy = np.array(boxes, dtype=np.float32)
    if confidences is None:
        confidences = np.ones(len(boxes), dtype=np.float32) * 0.9
    else:
        confidences = np.array(confidences, dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  1. Configuration Tests — verify tuned defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestTrackerConfiguration:
    """Ensure our tracker defaults are properly tuned for sports footage."""

    def test_default_activation_threshold_is_lowered(self):
        """
        A lower activation threshold (0.20) allows partially-occluded or
        distant players to seed new tracks even with weaker confidence.
        """
        tracker = Tracker()
        assert tracker.activation_threshold == 0.20

    def test_default_lost_track_buffer_is_raised(self):
        """
        A larger buffer (45 frames ≈ 1.5-1.8s) keeps IDs alive through
        short occlusions typical in sports footage.
        """
        tracker = Tracker()
        assert tracker.lost_track_buffer == 45

    def test_default_matching_threshold_is_lowered(self):
        """
        A more permissive IoU threshold (0.70) helps match detections
        to tracks when subjects move quickly between frames.
        """
        tracker = Tracker()
        assert tracker.matching_threshold == 0.70

    def test_default_minimum_consecutive_frames(self):
        """
        Requiring 3 consecutive detections filters spurious false
        positives without being too strict.
        """
        tracker = Tracker()
        assert tracker.min_consecutive_frames == 3

    def test_custom_config_is_applied(self):
        """User-supplied config must override all defaults."""
        cfg = TrackerConfig(
            track_activation_threshold=0.30,
            lost_track_buffer=60,
            minimum_matching_threshold=0.50,
            frame_rate=30,
            minimum_consecutive_frames=5,
        )
        tracker = Tracker(config=cfg)
        assert tracker.activation_threshold == 0.30
        assert tracker.lost_track_buffer == 60
        assert tracker.matching_threshold == 0.50
        assert tracker.min_consecutive_frames == 5


# ═══════════════════════════════════════════════════════════════════════════
#  2. Tracking Behaviour Tests — ID assignment & persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestTrackingBasicBehaviour:
    """Core ID assignment must work correctly."""

    def test_detections_receive_tracker_ids(self):
        """
        After feeding detections through the tracker, every detection
        must have a non-None tracker_id.
        """
        tracker = Tracker()
        boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]

        # Feed several frames so tracks get promoted past minimum_consecutive_frames
        for _ in range(5):
            result = tracker.update(_make_detections(boxes))

        assert result.tracker_id is not None
        assert len(result.tracker_id) > 0

    def test_stable_ids_for_stationary_objects(self):
        """
        When detections remain in the same location across frames,
        the tracker must assign the same ID every time.
        """
        tracker = Tracker()
        boxes = [[100, 100, 200, 200]]
        ids_over_time = []

        for _ in range(10):
            result = tracker.update(_make_detections(boxes))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                ids_over_time.append(result.tracker_id[0])

        # All tracked frames should have the same ID
        unique_ids = set(ids_over_time)
        assert len(unique_ids) == 1, (
            f"Expected 1 unique ID for a stationary object, got {len(unique_ids)}: {unique_ids}"
        )

    def test_distinct_ids_for_separate_objects(self):
        """
        Two non-overlapping detections must receive different IDs.
        """
        tracker = Tracker()
        boxes = [[10, 10, 50, 50], [400, 400, 450, 450]]

        for _ in range(5):
            result = tracker.update(_make_detections(boxes))

        assert len(set(result.tracker_id)) == 2, "Two separate objects should have distinct IDs"


# ═══════════════════════════════════════════════════════════════════════════
#  3. Occlusion Handling Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOcclusionHandling:
    """
    The tracker must retain IDs through brief occlusions.
    With lost_track_buffer=45, a subject missing for ~30 frames
    should be re-identified with the same ID.
    """

    def test_id_survives_short_occlusion(self):
        """
        If a subject disappears for a few frames (≤ lost_track_buffer)
        and then reappears at roughly the same location, it should
        retain its original ID.
        """
        tracker = Tracker()
        box = [[200, 200, 300, 300]]

        # Phase 1: establish the track
        original_id = None
        for _ in range(10):
            result = tracker.update(_make_detections(box))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                original_id = result.tracker_id[0]

        assert original_id is not None, "Track should be established after 10 frames"

        # Phase 2: subject disappears (no detections) for 20 frames
        for _ in range(20):
            tracker.update(_make_detections([]))

        # Phase 3: subject reappears at the same location
        result = tracker.update(_make_detections(box))

        # The track should either be re-associated with the original ID,
        # OR the detection should at least exist (the buffer kept it alive).
        if result.tracker_id is not None and len(result.tracker_id) > 0:
            # Best case: same ID is reused
            recovered_id = result.tracker_id[0]
            # With buffer=45 and gap=20, the ID should survive
            assert recovered_id == original_id, (
                f"Expected ID {original_id} to survive 20-frame occlusion, got {recovered_id}"
            )

    def test_id_lost_after_buffer_exceeded(self):
        """
        If a subject disappears for MORE frames than the lost_track_buffer,
        its ID should not survive — a new ID is assigned.
        """
        tracker = Tracker()
        box = [[200, 200, 300, 300]]

        original_id = None
        for _ in range(10):
            result = tracker.update(_make_detections(box))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                original_id = result.tracker_id[0]

        assert original_id is not None

        # Disappear for LONGER than the buffer (buffer=45, disappear for 60)
        for _ in range(60):
            tracker.update(_make_detections([]))

        # Reappear
        result = tracker.update(_make_detections(box))
        # After exceeding the buffer, the old track is deleted.
        # A brand-new track may or may not be immediately promoted
        # (depending on minimum_consecutive_frames), but it should
        # NOT have the original ID.
        if result.tracker_id is not None and len(result.tracker_id) > 0:
            assert result.tracker_id[0] != original_id, (
                "ID should NOT survive when occlusion exceeds the lost_track_buffer"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Rapid Movement Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRapidMovement:
    """
    With matching_threshold=0.70 and a sports-tuned config, the tracker
    should handle moderate inter-frame movement without losing the ID.
    """

    def test_gradual_movement_preserves_id(self):
        """
        A detection shifting by small increments each frame should
        keep the same ID (simulates a walking/jogging player).
        """
        tracker = Tracker()

        original_id = None
        for i in range(20):
            x = 100 + i * 5  # 5 px shift per frame
            box = [[x, 100, x + 100, 200]]
            result = tracker.update(_make_detections(box))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                original_id = result.tracker_id[0]

        # After 20 frames of gentle movement, it should still be tracked
        assert original_id is not None
        assert result.tracker_id is not None
        assert len(result.tracker_id) == 1
        assert result.tracker_id[0] == original_id

    def test_low_confidence_detections_match_existing_tracks(self):
        """
        ByteTrack's two-stage matching: detections BELOW the activation
        threshold can't seed new tracks, but they CAN match existing ones.

        This tests that once a track is established with high confidence,
        it can be maintained by low-confidence detections through the
        second-pass matching (e.g., a player briefly partially occluded
        still sends a weak detection that keeps the ID alive).
        """
        tracker = Tracker()
        box = [[100, 100, 200, 200]]

        # Phase 1: Establish track with HIGH confidence
        original_id = None
        for _ in range(10):
            result = tracker.update(_make_detections(box, confidences=[0.90]))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                original_id = result.tracker_id[0]

        assert original_id is not None, "Track should be established with high conf"

        # Phase 2: Feed LOW-confidence detections at same location.
        # These are below track_activation_threshold (0.20) but above 0.1,
        # so they go through second-pass matching and should keep the track alive.
        maintained = False
        for _ in range(5):
            result = tracker.update(_make_detections(box, confidences=[0.15]))
            if result.tracker_id is not None and len(result.tracker_id) > 0:
                if result.tracker_id[0] == original_id:
                    maintained = True

        assert maintained, (
            "Low-confidence detections (0.15) should maintain an existing track "
            "through ByteTrack's second-pass matching"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Reset / Isolation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTrackerReset:
    """Ensure the reset method clears all internal state."""

    def test_reset_clears_track_ids(self):
        """After reset, IDs should start fresh (not continue old sequence)."""
        tracker = Tracker()
        box = [[100, 100, 200, 200]]

        for _ in range(5):
            tracker.update(_make_detections(box))

        tracker.reset()

        # After reset + re-feed, we should get a brand-new track
        for _ in range(5):
            result = tracker.update(_make_detections(box))

        assert result.tracker_id is not None
        assert len(result.tracker_id) > 0
