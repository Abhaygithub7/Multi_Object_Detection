"""
video_downloader.py — YouTube video download utility.

Downloads a YouTube video via yt-dlp, extracts metadata (title, duration),
and returns the local file path for pipeline processing.
"""

import os
import json
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoMeta:
    """Metadata about a downloaded video."""
    title: str
    duration: float  # seconds
    filepath: str
    url: str


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr}")
    return json.loads(result.stdout)


def download_video(
    url: str,
    output_dir: str = "downloads",
    max_height: int = 720,
) -> VideoMeta:
    """
    Download a YouTube video at up to *max_height* resolution.

    Returns a VideoMeta with the title, duration, and local file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get metadata first
    info = get_video_info(url)
    title = info.get("title", "Unknown")
    duration = info.get("duration", 0)

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
    output_path = os.path.join(output_dir, f"{safe_title}.mp4")

    # Skip download if already exists
    if os.path.exists(output_path):
        return VideoMeta(
            title=title,
            duration=duration,
            filepath=output_path,
            url=url,
        )

    # Download
    fmt = f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]"
    result = subprocess.run(
        ["yt-dlp", "-f", fmt, "-o", output_path, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")

    return VideoMeta(
        title=title,
        duration=duration,
        filepath=output_path,
        url=url,
    )
