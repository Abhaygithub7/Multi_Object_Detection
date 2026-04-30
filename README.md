# Multi-Object Detection and Persistent ID Tracking

A modular computer vision pipeline for detecting and tracking multiple subjects in public sports/event footage with persistent unique IDs, trajectory visualization, team-colour clustering, and real-time object counting.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Video Source](#video-source)
- [Assumptions & Limitations](#assumptions--limitations)
- [Model & Tracker Choices](#model--tracker-choices)
- [Project Structure](#project-structure)

---

## Overview

This project implements a complete object detection and multi-object tracking (MOT) pipeline designed for sports footage. It uses **YOLOv8** for detection and **ByteTrack** for persistent ID assignment, with additional enhancements including trajectory trails, jersey-based team clustering, and a live object counter.

The pipeline processes a publicly available football video clip (20s–32s segment) and produces an annotated output video showing bounding boxes with unique IDs that remain consistent through occlusions, camera movement, and rapid player motion.

## Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | YOLOv8m detects all persons in each frame |
| **Persistent ID Tracking** | ByteTrack assigns stable IDs across frames, surviving brief occlusions |
| **Trajectory Visualization** | Fading 60-frame trails show recent movement paths |
| **Team Clustering** | K-Means clustering on HSV colour histograms auto-classifies players by jersey colour |
| **Object Count Overlay** | Live count of currently tracked subjects displayed on screen |
| **Modular Architecture** | Each component (detector, tracker, annotator, team classifier) is independently configurable and testable |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (Pipeline Orchestrator)                    │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐           │
│  │ detector │──▶│ tracker  │──▶│ team_classifier│           │
│  │ (YOLOv8) │   │(ByteTrack│   │ (K-Means HSV) │           │
│  └──────────┘   │  tuned)  │   └───────┬───────┘           │
│                 └──────────┘           │                    │
│                                        ▼                    │
│                              ┌──────────────────┐           │
│                              │    annotator      │           │
│                              │ (boxes, labels,   │           │
│                              │  traces, count)   │           │
│                              └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Predusk

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download YOLOv8 weights (automatic on first run, or manually)
# The yolov8m.pt model will be auto-downloaded by ultralytics on first use.
```

## Usage

### Process a Video Clip

```bash
# Basic usage — process a specific time segment
python main.py --source full_video.mp4 --output output_annotated.mp4 \
    --start_sec 20 --end_sec 32

# Disable team clustering
python main.py --source full_video.mp4 --output output.mp4 \
    --start_sec 20 --end_sec 32 --no-teams

# Process first 100 frames only (for quick testing)
python main.py --source full_video.mp4 --output test.mp4 --limit 100
```

### Capture Sample Screenshots

```bash
python screenshot.py --source full_video.mp4 --output_dir screenshots \
    --start_sec 20 --end_sec 32 --every 50
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | *required* | Path to input video |
| `--output` | str | `output.mp4` | Path for annotated output video |
| `--start_sec` | float | `None` | Start processing at this second |
| `--end_sec` | float | `None` | Stop processing at this second |
| `--limit` | int | `None` | Max number of frames to process |
| `--no-teams` | flag | `False` | Disable team colour clustering |

## Testing

The project follows **Test-Driven Development (TDD)** with 41 tests across 3 test files:

```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test modules
PYTHONPATH=. python -m pytest tests/test_tracker.py -v
PYTHONPATH=. python -m pytest tests/test_annotator.py -v
PYTHONPATH=. python -m pytest tests/test_team_classifier.py -v
```

### Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_tracker.py` | 13 | ByteTrack config, occlusion handling, rapid movement, low-conf matching |
| `test_annotator.py` | 16 | Bounding boxes, labels, traces, object count, feature toggling |
| `test_team_classifier.py` | 12 | HSV feature extraction, K-Means clustering, edge cases |

## Video Source

- **Source**: [YouTube — Football Match Footage](https://youtu.be/azLiQWUMp38?si=KMCm40d9-vmWNic-)
- **Segment Used**: 20s to 32s (12 seconds, 301 frames at 25 fps)
- **Content**: Football/soccer match with multiple players, camera panning, and player overlaps

## Assumptions & Limitations

### Assumptions

- The video contains clearly visible human subjects (players) as the primary tracking targets
- COCO class `0` (person) is sufficient for detection — no sport-specific fine-tuning required
- Two-team clustering is appropriate for the sports footage being processed
- The source video is publicly accessible

### Limitations

- **Resolution**: Processing at 640×360 (source resolution) — higher resolution would improve small-player detection but increase processing time
- **Team Clustering Accuracy**: K-Means on HSV histograms may struggle with similar jersey colours, monochrome kits, or when referees are present (they may form a "third cluster")
- **ID Switches**: Despite tuned ByteTrack parameters, rapid close-proximity crossovers between players can still cause occasional ID switches
- **No Re-ID Model**: The tracker uses IoU-based association only — a dedicated ReID model (e.g., OSNet) would improve long-term ID consistency after extended occlusions
- **Camera Motion**: ByteTrack does not include explicit camera motion compensation — BoT-SORT would be needed for extreme pan/zoom scenarios

## Model & Tracker Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Detector** | YOLOv8m | Best speed/accuracy tradeoff; pre-trained on COCO with strong person detection |
| **Tracker** | ByteTrack (via supervision) | Two-stage matching allows low-conf detections to maintain existing tracks; Kalman filter provides motion prediction |
| **Team Classifier** | K-Means on HSV histograms | Lightweight, unsupervised — no labelled data needed; HSV is robust to lighting variation |
| **Annotations** | Roboflow Supervision | Production-quality annotators with built-in colour palettes and trace rendering |

### ByteTrack Tuning for Sports

| Parameter | Default → Tuned | Rationale |
|-----------|-----------------|-----------|
| `track_activation_threshold` | 0.25 → 0.20 | Allows partially-occluded/distant players to seed tracks |
| `lost_track_buffer` | 30 → 45 | Retains IDs through ~1.5s occlusions |
| `minimum_matching_threshold` | 0.8 → 0.70 | More permissive IoU for fast-moving players |
| `minimum_consecutive_frames` | 1 → 3 | Filters spurious one-frame false positives |

## Project Structure

```
Predusk/
├── main.py                  # Pipeline orchestrator
├── detector.py              # YOLOv8 detection module
├── tracker.py               # ByteTrack tracking module (tuned config)
├── annotator.py             # Frame annotation (boxes, labels, traces, count)
├── team_classifier.py       # K-Means team clustering on HSV histograms
├── screenshot.py            # Screenshot capture utility
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── Problem_Statement.md     # Original assignment brief
├── TECHNICAL_REPORT.md      # Detailed technical report
├── output_annotated.mp4     # Final annotated output video
├── screenshots/             # Sample result screenshots
│   ├── frame_00550.png
│   ├── frame_00600.png
│   └── ...
└── tests/
    ├── test_tracker.py      # 13 tests — tracker behaviour
    ├── test_annotator.py    # 16 tests — annotation features
    └── test_team_classifier.py  # 12 tests — team clustering
```

---

## License

This project was created as part of a technical assessment. The source video is publicly available footage used for educational/evaluation purposes only.
