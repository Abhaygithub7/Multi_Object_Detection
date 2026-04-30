"""
team_classifier.py — Team / Role Clustering module.

Classifies bounding-box crops into N teams using colour histograms
in HSV space and K-Means clustering.  This enables colour-coded
annotation of different teams on the field.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans


@dataclass
class TeamClassifierConfig:
    """
    Configuration for team clustering.

    Attributes:
        n_teams:      Number of teams to classify into (default 2).
        color_space:  Colour space for histogram extraction ('hsv' or 'lab').
        h_bins:       Number of histogram bins for the Hue channel.
        s_bins:       Number of histogram bins for the Saturation channel.
    """
    n_teams: int = 2
    color_space: str = "hsv"
    h_bins: int = 30
    s_bins: int = 32


class TeamClassifier:
    """
    Clusters bounding-box crops into teams using colour histograms
    and K-Means.
    """

    def __init__(self, config: Optional[TeamClassifierConfig] = None):
        self.config = config or TeamClassifierConfig()
        self._kmeans: Optional[KMeans] = None
        self._cluster_centers_bgr: Dict[int, Tuple[int, int, int]] = {}

    # ── Feature extraction ──────────────────────────────────────────

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a normalised 2-D colour histogram (H, S) from *crop*
        and return it as a flattened 1-D feature vector.

        Using HS (ignoring V) makes the features more robust to
        lighting / shadow differences on the field.
        """
        if self.config.color_space == "hsv":
            converted = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        elif self.config.color_space == "lab":
            converted = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        else:
            converted = crop

        # 2-D histogram over channels 0 and 1 (H+S in HSV, L+a in LAB)
        hist = cv2.calcHist(
            [converted],
            [0, 1],
            None,
            [self.config.h_bins, self.config.s_bins],
            [0, 180, 0, 256] if self.config.color_space == "hsv" else [0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    # ── Classification ──────────────────────────────────────────────

    def fit_predict(self, crops: List[np.ndarray]) -> List[int]:
        """
        Cluster *crops* into teams and return a list of integer labels
        (one per crop).

        Also stores cluster centres so that get_team_colors() can be
        called afterwards.
        """
        if len(crops) == 0:
            return []

        features = np.array([self.extract_features(c) for c in crops])

        n_clusters = min(self.config.n_teams, len(crops))
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
        )
        labels = self._kmeans.fit_predict(features).tolist()

        # Compute the average BGR colour per cluster for colour-coded boxes
        self._cluster_centers_bgr = {}
        for label in set(labels):
            cluster_crops = [c for c, l in zip(crops, labels) if l == label]
            avg_colors = [c.mean(axis=(0, 1)) for c in cluster_crops]
            avg = np.mean(avg_colors, axis=0).astype(int)
            self._cluster_centers_bgr[label] = (int(avg[0]), int(avg[1]), int(avg[2]))

        return labels

    def predict(self, crop: np.ndarray) -> int:
        """Predict the team label for a single new crop (after fitting)."""
        if self._kmeans is None:
            raise RuntimeError("Call fit_predict() before predict().")
        feat = self.extract_features(crop).reshape(1, -1)
        return int(self._kmeans.predict(feat)[0])

    # ── Accessors ───────────────────────────────────────────────────

    def get_team_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Return a dict mapping each team label to its representative
        BGR colour (averaged from the cluster's crops).
        """
        return self._cluster_centers_bgr
