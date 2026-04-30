"""
tests/test_team_classifier.py — TDD tests for Phase 2, Feature 4:
    Team / Role Clustering

Uses colour histograms + K-Means to classify bounding-box crops into
two teams based on dominant jersey colour.
"""

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _solid_crop(color_bgr, h=100, w=60):
    """Create a solid-colour crop in BGR."""
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:] = color_bgr
    return crop


def _noisy_crop(base_color_bgr, h=100, w=60, noise_level=15):
    """Create a crop with Gaussian noise around a base colour."""
    crop = np.full((h, w, 3), base_color_bgr, dtype=np.float32)
    noise = np.random.normal(0, noise_level, crop.shape)
    crop = np.clip(crop + noise, 0, 255).astype(np.uint8)
    return crop


# ═══════════════════════════════════════════════════════════════════════════
#  1. Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTeamClassifierConfig:
    """Validate the default configuration."""

    def test_default_n_clusters(self):
        """Should default to 2 teams."""
        from team_classifier import TeamClassifierConfig
        cfg = TeamClassifierConfig()
        assert cfg.n_teams == 2

    def test_default_colour_space(self):
        """Should use HSV for better colour separation."""
        from team_classifier import TeamClassifierConfig
        cfg = TeamClassifierConfig()
        assert cfg.color_space == "hsv"

    def test_custom_config(self):
        """Custom config should be respected."""
        from team_classifier import TeamClassifierConfig
        cfg = TeamClassifierConfig(n_teams=3, color_space="lab")
        assert cfg.n_teams == 3
        assert cfg.color_space == "lab"


# ═══════════════════════════════════════════════════════════════════════════
#  2. Feature Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Colour histogram extraction must produce valid feature vectors."""

    def test_extract_features_returns_correct_shape(self):
        """
        For a single crop, extract_features should return a 1-D
        feature vector (histogram).
        """
        from team_classifier import TeamClassifier
        tc = TeamClassifier()
        crop = _solid_crop((255, 0, 0))  # Pure blue in BGR
        features = tc.extract_features(crop)
        assert features.ndim == 1
        assert len(features) > 0

    def test_different_colours_produce_different_features(self):
        """Red and blue crops must yield distinct feature vectors."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()
        red_crop = _solid_crop((0, 0, 255))   # Red in BGR
        blue_crop = _solid_crop((255, 0, 0))  # Blue in BGR
        feat_red = tc.extract_features(red_crop)
        feat_blue = tc.extract_features(blue_crop)
        assert not np.allclose(feat_red, feat_blue), (
            "Feature vectors for red and blue should be different"
        )

    def test_similar_colours_produce_similar_features(self):
        """Two slightly-different shades of red should be closer together."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()
        red1 = _solid_crop((0, 0, 200))
        red2 = _solid_crop((0, 0, 220))
        blue = _solid_crop((255, 0, 0))

        feat_r1 = tc.extract_features(red1)
        feat_r2 = tc.extract_features(red2)
        feat_b  = tc.extract_features(blue)

        dist_reds = np.linalg.norm(feat_r1 - feat_r2)
        dist_rb   = np.linalg.norm(feat_r1 - feat_b)

        assert dist_reds < dist_rb, (
            "Two shades of red should be closer than red vs blue"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Classification (K-Means Clustering) Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTeamClassification:
    """K-Means must correctly separate two clearly different team colours."""

    def test_classify_two_teams(self):
        """
        Given 6 crops — 3 red, 3 blue — the classifier must assign
        exactly 2 distinct team labels.
        """
        from team_classifier import TeamClassifier
        tc = TeamClassifier()

        crops = [
            _noisy_crop((0, 0, 200)),   # Red-ish
            _noisy_crop((0, 0, 210)),   # Red-ish
            _noisy_crop((0, 0, 190)),   # Red-ish
            _noisy_crop((200, 0, 0)),   # Blue-ish
            _noisy_crop((210, 0, 0)),   # Blue-ish
            _noisy_crop((190, 0, 0)),   # Blue-ish
        ]

        labels = tc.fit_predict(crops)
        assert len(labels) == 6
        assert len(set(labels)) == 2, (
            f"Expected 2 teams, got {len(set(labels))}: {labels}"
        )

    def test_same_team_gets_same_label(self):
        """All red crops should share the same label."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()

        crops = [
            _noisy_crop((0, 0, 200)),   # Red
            _noisy_crop((0, 0, 210)),   # Red
            _noisy_crop((0, 0, 195)),   # Red
            _noisy_crop((200, 0, 0)),   # Blue
            _noisy_crop((210, 0, 0)),   # Blue
            _noisy_crop((195, 0, 0)),   # Blue
        ]

        labels = tc.fit_predict(crops)
        # First 3 (red) should share a label, last 3 (blue) another
        assert labels[0] == labels[1] == labels[2], (
            f"Red crops should share a label: {labels[:3]}"
        )
        assert labels[3] == labels[4] == labels[5], (
            f"Blue crops should share a label: {labels[3:]}"
        )
        assert labels[0] != labels[3], "Red and blue should get different labels"

    def test_classify_returns_int_labels(self):
        """Labels must be integers (for use as colour-map keys)."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()

        crops = [_noisy_crop((0, 0, 200)), _noisy_crop((200, 0, 0))]
        labels = tc.fit_predict(crops)
        for lbl in labels:
            assert isinstance(lbl, (int, np.integer))

    def test_get_team_colors_returns_bgr_tuples(self):
        """
        After fitting, get_team_colors should return a dict
        mapping each team label to a representative BGR tuple.
        """
        from team_classifier import TeamClassifier
        tc = TeamClassifier()

        crops = [
            _solid_crop((0, 0, 200)),   # Red
            _solid_crop((200, 0, 0)),   # Blue
        ]
        tc.fit_predict(crops)
        colors = tc.get_team_colors()

        assert isinstance(colors, dict)
        assert len(colors) == 2
        for label, bgr in colors.items():
            assert len(bgr) == 3, f"BGR tuple must have 3 values, got {bgr}"

    def test_single_crop_still_works(self):
        """Edge case: only one crop should not crash (assigns team 0)."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()
        crops = [_solid_crop((0, 0, 200))]
        labels = tc.fit_predict(crops)
        assert len(labels) == 1

    def test_empty_crops_returns_empty(self):
        """Edge case: no crops should return empty list."""
        from team_classifier import TeamClassifier
        tc = TeamClassifier()
        labels = tc.fit_predict([])
        assert len(labels) == 0
