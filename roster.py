"""
roster.py — Roster management module.

Loads match roster from a JSON file and provides team-aware label
generation: mapping (team_label, jersey_number) → "Player Name #N".
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TeamInfo:
    """Holds information about a single team."""
    name: str
    short: str
    kit_color_desc: str
    box_color_bgr: Tuple[int, int, int]
    players: Dict[str, str]  # jersey_number_str → player_name


@dataclass
class RosterData:
    """Parsed roster for a match."""
    match: str
    team_a: TeamInfo
    team_b: TeamInfo
    referee_color_bgr: Tuple[int, int, int] = (0, 255, 0)


class RosterManager:
    """
    Loads roster from JSON and provides team-colour mapping and
    player-name lookup.
    """

    def __init__(self, roster_path: str):
        self.roster_path = roster_path
        self.data = self._load(roster_path)

        # After team clustering, we map cluster_label → "team_a" or "team_b"
        self._cluster_to_team: Dict[int, str] = {}

    # ── Loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load(path: str) -> RosterData:
        """Parse roster.json into a RosterData object."""
        with open(path, "r") as f:
            raw = json.load(f)

        team_a = TeamInfo(
            name=raw["team_a"]["name"],
            short=raw["team_a"]["short"],
            kit_color_desc=raw["team_a"]["kit_color_desc"],
            box_color_bgr=tuple(raw["team_a"]["box_color_bgr"]),
            players=raw["team_a"]["players"],
        )
        team_b = TeamInfo(
            name=raw["team_b"]["name"],
            short=raw["team_b"]["short"],
            kit_color_desc=raw["team_b"]["kit_color_desc"],
            box_color_bgr=tuple(raw["team_b"]["box_color_bgr"]),
            players=raw["team_b"]["players"],
        )
        ref_color = tuple(raw.get("referee", {}).get("box_color_bgr", [0, 255, 0]))

        return RosterData(
            match=raw.get("match", ""),
            team_a=team_a,
            team_b=team_b,
            referee_color_bgr=ref_color,
        )

    # ── Cluster-to-team mapping ─────────────────────────────────────

    def map_clusters_to_teams(
        self, cluster_avg_colors_bgr: Dict[int, Tuple[int, int, int]]
    ):
        """
        Given the average BGR colour per cluster from the TeamClassifier,
        determine which cluster corresponds to team_a (lighter kit / white)
        and which to team_b (darker / coloured kit).

        Heuristic: Real Madrid wears white → higher average brightness.
        Dortmund wears yellow/black → relatively lower brightness or
        stronger saturation.

        We use the V (value/brightness) channel in HSV to distinguish.
        """
        if len(cluster_avg_colors_bgr) < 2:
            # Fallback: just assign in order
            for i, label in enumerate(sorted(cluster_avg_colors_bgr.keys())):
                self._cluster_to_team[label] = "team_a" if i == 0 else "team_b"
            return

        # Convert each cluster's average BGR to HSV and compare brightness
        brightness = {}
        for label, bgr in cluster_avg_colors_bgr.items():
            pixel = np.array([[bgr]], dtype=np.uint8)
            hsv = __import__("cv2").cvtColor(pixel, __import__("cv2").COLOR_BGR2HSV)
            brightness[label] = int(hsv[0, 0, 2])  # V channel

        # Sort by brightness: brightest cluster → team_a (white kit)
        sorted_labels = sorted(brightness.keys(), key=lambda l: brightness[l], reverse=True)
        self._cluster_to_team[sorted_labels[0]] = "team_a"
        self._cluster_to_team[sorted_labels[1]] = "team_b"

    # ── Accessors ───────────────────────────────────────────────────

    def get_team_for_cluster(self, cluster_label: int) -> str:
        """Return 'team_a' or 'team_b' for a given cluster label."""
        return self._cluster_to_team.get(cluster_label, "team_a")

    def get_team_info(self, team_key: str) -> TeamInfo:
        """Return TeamInfo for 'team_a' or 'team_b'."""
        if team_key == "team_a":
            return self.data.team_a
        return self.data.team_b

    def get_box_color_bgr(self, cluster_label: int) -> Tuple[int, int, int]:
        """Get the configured box colour for a cluster's team."""
        team_key = self.get_team_for_cluster(cluster_label)
        info = self.get_team_info(team_key)
        return info.box_color_bgr

    def get_team_name(self, cluster_label: int) -> str:
        """Get the short team name for a cluster."""
        team_key = self.get_team_for_cluster(cluster_label)
        return self.get_team_info(team_key).short

    def lookup_player(self, cluster_label: int, jersey_number: str) -> Optional[str]:
        """
        Look up a player name by cluster label and jersey number.
        Returns None if not found.
        """
        team_key = self.get_team_for_cluster(cluster_label)
        info = self.get_team_info(team_key)
        return info.players.get(jersey_number)

    def build_label(
        self,
        cluster_label: int,
        tracker_id: int,
        jersey_number: Optional[str] = None,
    ) -> str:
        """
        Build a display label for a player.

        If jersey_number is known → "Player Name #N [TEAM]"
        Otherwise → "#tracker_id [TEAM]"
        """
        team_short = self.get_team_name(cluster_label)

        if jersey_number:
            player = self.lookup_player(cluster_label, jersey_number)
            if player:
                return f"{player} #{jersey_number} [{team_short}]"
            return f"#{jersey_number} [{team_short}]"

        return f"#{tracker_id} [{team_short}]"

    def get_all_team_colors_sv(self) -> Dict[int, "sv.Color"]:
        """Return a mapping of cluster_label → sv.Color for use in annotators."""
        import supervision as sv
        colors = {}
        for label, team_key in self._cluster_to_team.items():
            bgr = self.get_box_color_bgr(label)
            colors[label] = sv.Color(r=int(bgr[2]), g=int(bgr[1]), b=int(bgr[0]))
        return colors
