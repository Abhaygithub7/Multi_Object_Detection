"""
main.py — Multi-Object Detection and Persistent ID Tracking pipeline.

Orchestrates the Detector, Tracker, FrameAnnotator, TeamClassifier,
and RosterManager to process video footage, producing an annotated
output with:
  • Team-coloured bounding boxes (e.g. Real Madrid = cyan, BVB = yellow)
  • Player name labels when jersey numbers are known
  • Trajectory traces (fading trails)
  • Live object count overlay
"""

import cv2
import argparse
import numpy as np
import supervision as sv

from detector import Detector, DetectorConfig
from tracker import Tracker, TrackerConfig
from annotator import FrameAnnotator, AnnotatorConfig
from team_classifier import TeamClassifier, TeamClassifierConfig
from roster import RosterManager


def _extract_crops(frame: np.ndarray, detections: sv.Detections):
    """Extract bounding-box crops from the frame for each detection."""
    crops = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
        else:
            crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
    return crops


def process_video(
    source_video_path: str,
    output_video_path: str,
    start_sec: float = None,
    end_sec: float = None,
    limit: int = None,
    enable_team_clustering: bool = True,
    team_calibration_frames: int = 30,
    roster_path: str = None,
):
    print(f"Processing video: {source_video_path}")

    # ── Initialise pipeline components ──────────────────────────────
    detector = Detector(DetectorConfig())
    tracker = Tracker(TrackerConfig())
    annotator = FrameAnnotator(AnnotatorConfig(
        box_thickness=2,
        label_text_scale=0.5,
    ))
    team_clf = TeamClassifier(TeamClassifierConfig()) if enable_team_clustering else None

    # ── Roster (optional) ───────────────────────────────────────────
    roster = None
    if roster_path:
        roster = RosterManager(roster_path)
        print(f"Roster loaded: {roster.data.match}")
        print(f"  Team A: {roster.data.team_a.name} ({roster.data.team_a.kit_color_desc})")
        print(f"  Team B: {roster.data.team_b.name} ({roster.data.team_b.kit_color_desc})")

    # ── Video metadata ──────────────────────────────────────────────
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    print(
        f"Video Info: {video_info.width}x{video_info.height}, "
        f"{video_info.fps} fps, {video_info.total_frames} frames"
    )

    tracker.config.frame_rate = int(video_info.fps)
    generator = sv.get_video_frames_generator(source_video_path)

    # ── Frame range ─────────────────────────────────────────────────
    start_frame = int(start_sec * video_info.fps) if start_sec is not None else 0
    end_frame = int(end_sec * video_info.fps) if end_sec is not None else float("inf")

    # ── Team calibration state ──────────────────────────────────────
    calibration_crops = []
    team_colors_bgr = {}
    team_fitted = False

    # ── Processing loop ─────────────────────────────────────────────
    frame_count = 0
    processed_count = 0
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
        for frame in generator:
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count > end_frame or (
                limit and frame_count - start_frame >= limit
            ):
                break

            frame_count += 1
            processed_count += 1

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            detections = tracker.update(detections)

            # 3. Team clustering
            team_labels_list = None
            custom_labels = None

            if team_clf is not None and len(detections) > 0:
                crops = _extract_crops(frame, detections)

                if not team_fitted:
                    calibration_crops.extend(crops)
                    if processed_count >= team_calibration_frames and len(calibration_crops) >= 4:
                        team_clf.fit_predict(calibration_crops)
                        team_colors_bgr = team_clf.get_team_colors()
                        team_fitted = True

                        # Map clusters to real teams if roster is loaded
                        if roster:
                            roster.map_clusters_to_teams(team_colors_bgr)
                            # Use roster-defined box colours
                            mapped_colors = {}
                            for cl in team_colors_bgr:
                                mapped_colors[cl] = roster.get_box_color_bgr(cl)
                            team_colors_bgr = mapped_colors

                            print(
                                f"Team clustering mapped: "
                                f"cluster 0 → {roster.get_team_name(0)}, "
                                f"cluster 1 → {roster.get_team_name(1)}"
                            )
                        else:
                            print(f"Team clustering calibrated. Colors: {team_colors_bgr}")

                        annotator.set_team_colors(team_colors_bgr)

                if team_fitted:
                    team_labels_list = [team_clf.predict(c) for c in crops]
                    detections.class_id = np.array(team_labels_list, dtype=int)

                    # Build labels with roster names
                    if roster:
                        custom_labels = [
                            roster.build_label(tl, tid)
                            for tl, tid in zip(team_labels_list, detections.tracker_id)
                        ]

            # 4. Annotate
            out = annotator.annotate(
                frame, detections,
                custom_labels=custom_labels,
                team_labels=team_labels_list,
            )

            sink.write_frame(frame=out)

    print(f"Finished processing. Output saved to {output_video_path}")
    print(f"Processed {processed_count} frames.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection and Persistent ID Tracking"
    )
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--output", type=str, default="output.mp4", help="Path to output video"
    )
    parser.add_argument(
        "--start_sec", type=float, default=None, help="Start processing at this second"
    )
    parser.add_argument(
        "--end_sec", type=float, default=None, help="End processing at this second"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of frames to process"
    )
    parser.add_argument(
        "--no-teams", action="store_true", help="Disable team clustering"
    )
    parser.add_argument(
        "--roster", type=str, default=None,
        help="Path to roster JSON file for team names and player labels"
    )

    args = parser.parse_args()
    process_video(
        args.source,
        args.output,
        args.start_sec,
        args.end_sec,
        args.limit,
        enable_team_clustering=not args.no_teams,
        roster_path=args.roster,
    )
