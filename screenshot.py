"""
screenshot.py — Capture sample screenshots from the annotated pipeline.

Runs the full pipeline on a video and saves annotated frames as PNG images
at configurable intervals. Useful for generating submission screenshots.
"""

import cv2
import os
import numpy as np
import supervision as sv

from detector import Detector, DetectorConfig
from tracker import Tracker, TrackerConfig
from annotator import FrameAnnotator, AnnotatorConfig
from team_classifier import TeamClassifier, TeamClassifierConfig


def _extract_crops(frame, detections):
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


def capture_screenshots(
    source_video_path: str,
    output_dir: str = "screenshots",
    start_sec: float = None,
    end_sec: float = None,
    capture_every_n_frames: int = 25,
    team_calibration_frames: int = 30,
):
    """Run the pipeline and save annotated frames as screenshots."""
    os.makedirs(output_dir, exist_ok=True)

    detector = Detector(DetectorConfig())
    tracker = Tracker(TrackerConfig())
    annotator = FrameAnnotator(AnnotatorConfig())
    team_clf = TeamClassifier(TeamClassifierConfig())

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    tracker.config.frame_rate = int(video_info.fps)
    generator = sv.get_video_frames_generator(source_video_path)

    start_frame = int(start_sec * video_info.fps) if start_sec is not None else 0
    end_frame = int(end_sec * video_info.fps) if end_sec is not None else float("inf")

    calibration_crops = []
    team_colors = {}
    team_fitted = False

    frame_count = 0
    processed_count = 0
    saved_count = 0

    for frame in generator:
        if frame_count < start_frame:
            frame_count += 1
            continue
        if frame_count > end_frame:
            break

        frame_count += 1
        processed_count += 1

        detections = detector.detect(frame)
        detections = tracker.update(detections)

        # Team clustering
        if len(detections) > 0:
            crops = _extract_crops(frame, detections)
            if not team_fitted:
                calibration_crops.extend(crops)
                if processed_count >= team_calibration_frames and len(calibration_crops) >= 4:
                    team_clf.fit_predict(calibration_crops)
                    team_colors = team_clf.get_team_colors()
                    team_fitted = True
            if team_fitted:
                team_labels = [team_clf.predict(c) for c in crops]
                detections.class_id = np.array(team_labels, dtype=int)

        out = annotator.annotate(frame, detections)

        # Save screenshot at interval
        if processed_count % capture_every_n_frames == 0:
            path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(path, out)
            saved_count += 1
            print(f"Saved screenshot: {path}")

    print(f"Done. Saved {saved_count} screenshots to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture annotated screenshots")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="screenshots")
    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--every", type=int, default=25, help="Capture every N frames")

    args = parser.parse_args()
    capture_screenshots(
        args.source, args.output_dir, args.start_sec, args.end_sec, args.every,
    )
