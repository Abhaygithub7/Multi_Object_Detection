"""
detector.py — Object detection module.

Wraps YOLOv8 inference so it can be tested and configured independently.
"""

from ultralytics import YOLO
import supervision as sv
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DetectorConfig:
    """Configuration for the YOLOv8 detector."""
    model_name: str = "yolov8m.pt"
    target_classes: List[int] = field(default_factory=lambda: [0])  # COCO class 0 = person
    confidence_threshold: float = 0.25


class Detector:
    """Wraps YOLOv8 for frame-level object detection."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model = YOLO(self.config.model_name)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a single frame and return supervision Detections."""
        results = self.model(
            frame,
            classes=self.config.target_classes,
            conf=self.config.confidence_threshold,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)
