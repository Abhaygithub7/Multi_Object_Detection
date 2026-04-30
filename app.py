"""
app.py — Streamlit UI for Multi-Object Detection & Tracking.

Drop a YouTube URL → auto-download → auto-detect teams → run pipeline →
view annotated video with team-coloured bounding boxes and player labels.
"""

import os
import sys
import json
import time
import streamlit as st
import numpy as np
import cv2
import supervision as sv

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from video_downloader import download_video, get_video_info
from roster_generator import (
    auto_generate_roster_from_title,
    generate_roster,
    get_available_teams,
    get_team_key_by_name,
    TEAM_DB,
)
from detector import Detector, DetectorConfig
from tracker import Tracker, TrackerConfig
from annotator import FrameAnnotator, AnnotatorConfig
from team_classifier import TeamClassifier, TeamClassifierConfig
from roster import RosterManager


# ═══════════════════════════════════════════════════════════════════════════
#  Page Config & Styling
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sports Tracking Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.65);
        font-size: 1rem;
        margin: 0.4rem 0 0;
    }

    .status-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .status-card h3 {
        color: #00d4ff;
        margin: 0 0 0.3rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-card p {
        color: #fff;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .team-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }

    .sidebar .sidebar-content {
        background: #0f0c29;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
    }

    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }
    .step-num {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-text {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>⚽ Sports Tracking Engine</h1>
    <p>Drop a YouTube URL → auto-detect teams → get annotated video with player tracking</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar — Input Controls
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎬 Video Source")
    url = st.text_input(
        "YouTube URL",
        placeholder="https://youtu.be/...",
        help="Paste any YouTube football video URL",
    )

    st.markdown("---")
    st.markdown("### ⏱️ Time Range")
    col1, col2 = st.columns(2)
    with col1:
        start_sec = st.number_input("Start (sec)", min_value=0, value=20, step=1)
    with col2:
        end_sec = st.number_input("End (sec)", min_value=1, value=32, step=1)

    st.markdown("---")
    st.markdown("### 🏟️ Team Configuration")
    team_mode = st.radio(
        "How to set teams?",
        ["🔍 Auto-detect from video title", "✏️ Manual selection"],
        index=0,
    )

    team_a_key = None
    team_b_key = None

    if "✏️" in team_mode:
        available = get_available_teams()
        team_a_name = st.selectbox("Team A (home)", available, index=available.index("Real Madrid") if "Real Madrid" in available else 0)
        team_b_name = st.selectbox("Team B (away)", available, index=available.index("Bayern Munich") if "Bayern Munich" in available else 1)
        team_a_key = get_team_key_by_name(team_a_name)
        team_b_key = get_team_key_by_name(team_b_name)

    st.markdown("---")
    st.markdown("### ⚙️ Advanced")
    max_resolution = st.selectbox("Max Resolution", [360, 480, 720], index=2)
    calibration_frames = st.slider("Team calibration frames", 10, 60, 30)

    st.markdown("---")
    run_button = st.button("🚀 Run Pipeline", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════

def _extract_crops(frame, detections):
    crops = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        c = frame[y1:y2, x1:x2]
        crops.append(c if c.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))
    return crops


def run_pipeline(video_path, output_path, start, end, roster_path, calib_frames, progress_bar, status_text):
    """Run the full detection + tracking + annotation pipeline."""
    detector = Detector(DetectorConfig())
    tracker_ = Tracker(TrackerConfig())
    annotator_ = FrameAnnotator(AnnotatorConfig(box_thickness=2, label_text_scale=0.5))
    team_clf = TeamClassifier(TeamClassifierConfig())
    roster_mgr = RosterManager(roster_path) if roster_path else None

    video_info = sv.VideoInfo.from_video_path(video_path)
    tracker_.config.frame_rate = int(video_info.fps)
    generator = sv.get_video_frames_generator(video_path)

    start_frame = int(start * video_info.fps)
    end_frame = int(end * video_info.fps)
    total_frames = end_frame - start_frame

    calibration_crops = []
    team_fitted = False
    team_colors_bgr = {}
    screenshots = []

    frame_count = 0
    processed = 0

    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for frame in generator:
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count > end_frame:
                break

            frame_count += 1
            processed += 1

            # Progress
            pct = min(processed / max(total_frames, 1), 1.0)
            progress_bar.progress(pct)
            status_text.text(f"Processing frame {processed}/{total_frames}...")

            # Detect + Track
            dets = detector.detect(frame)
            dets = tracker_.update(dets)

            team_labels_list = None
            custom_labels = None

            if len(dets) > 0:
                crops = _extract_crops(frame, dets)

                if not team_fitted:
                    calibration_crops.extend(crops)
                    if processed >= calib_frames and len(calibration_crops) >= 4:
                        team_clf.fit_predict(calibration_crops)
                        team_colors_bgr = team_clf.get_team_colors()
                        team_fitted = True

                        if roster_mgr:
                            roster_mgr.map_clusters_to_teams(team_colors_bgr)
                            mapped = {cl: roster_mgr.get_box_color_bgr(cl) for cl in team_colors_bgr}
                            team_colors_bgr = mapped

                        annotator_.set_team_colors(team_colors_bgr)

                if team_fitted:
                    team_labels_list = [team_clf.predict(c) for c in crops]
                    dets.class_id = np.array(team_labels_list, dtype=int)

                    if roster_mgr:
                        custom_labels = [
                            roster_mgr.build_label(tl, tid)
                            for tl, tid in zip(team_labels_list, dets.tracker_id)
                        ]

            out = annotator_.annotate(frame, dets, custom_labels=custom_labels, team_labels=team_labels_list)
            sink.write_frame(frame=out)

            # Capture screenshots at intervals
            if processed in [1, total_frames // 4, total_frames // 2, 3 * total_frames // 4]:
                screenshots.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    return screenshots, processed


# ═══════════════════════════════════════════════════════════════════════════
#  Main Flow
# ═══════════════════════════════════════════════════════════════════════════

if run_button and url:
    # ── Step 1: Download Video ──────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📥 Step 1: Downloading Video")
        with st.spinner("Fetching video from YouTube..."):
            try:
                meta = download_video(url, output_dir="downloads", max_height=max_resolution)
                st.success(f"✅ Downloaded: **{meta.title}**")
                st.caption(f"Duration: {meta.duration:.0f}s | Path: `{meta.filepath}`")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                st.stop()

    # ── Step 2: Detect Teams & Generate Roster ──────────────────────
    with col_left:
        st.markdown("### 🏟️ Step 2: Team Detection & Roster")

    roster_path = None

    if "🔍" in team_mode:
        # Auto-detect from title
        ta_key, tb_key, auto_path = auto_generate_roster_from_title(
            meta.title, output_path="roster_auto.json"
        )
        if auto_path:
            roster_path = auto_path
            ta_info = TEAM_DB[ta_key]
            tb_info = TEAM_DB[tb_key]
            with col_left:
                st.success(f"✅ Auto-detected: **{ta_info['name']}** vs **{tb_info['name']}**")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="status-card">
                        <h3>Home Team</h3>
                        <p>{ta_info['name']} ({ta_info['short']})</p>
                        <small style="color:rgba(255,255,255,0.5)">Kit: {ta_info['kit_color_desc']} | {len(ta_info['players'])} players</small>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="status-card">
                        <h3>Away Team</h3>
                        <p>{tb_info['name']} ({tb_info['short']})</p>
                        <small style="color:rgba(255,255,255,0.5)">Kit: {tb_info['kit_color_desc']} | {len(tb_info['players'])} players</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with col_left:
                st.warning("⚠️ Couldn't auto-detect teams from the video title. Please select manually.")
                available = get_available_teams()
                mc1, mc2 = st.columns(2)
                with mc1:
                    team_a_name = st.selectbox("Team A", available, key="fallback_a")
                with mc2:
                    team_b_name = st.selectbox("Team B", available, key="fallback_b")
                team_a_key = get_team_key_by_name(team_a_name)
                team_b_key = get_team_key_by_name(team_b_name)
    else:
        pass  # Manual keys already set from sidebar

    if not roster_path and team_a_key and team_b_key:
        roster_path = generate_roster(team_a_key, team_b_key, meta.title, "roster_auto.json")
        ta_info = TEAM_DB[team_a_key]
        tb_info = TEAM_DB[team_b_key]
        with col_left:
            st.success(f"✅ Roster generated: **{ta_info['name']}** vs **{tb_info['name']}**")

    # Show roster JSON in expander
    if roster_path and os.path.exists(roster_path):
        with col_right:
            with st.expander("📋 View Roster JSON", expanded=False):
                with open(roster_path) as f:
                    st.json(json.load(f))

    # ── Step 3: Run Pipeline ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Step 3: Running Detection Pipeline")

    output_path = "output_ui.mp4"
    progress = st.progress(0)
    status = st.empty()

    t_start = time.time()
    screenshots, n_frames = run_pipeline(
        meta.filepath, output_path,
        start_sec, end_sec,
        roster_path, calibration_frames,
        progress, status,
    )
    elapsed = time.time() - t_start

    progress.progress(1.0)
    status.text("")

    # ── Step 4: Results ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎉 Results")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames Processed", f"{n_frames}")
    m2.metric("Processing Time", f"{elapsed:.1f}s")
    m3.metric("FPS", f"{n_frames / max(elapsed, 0.1):.1f}")
    m4.metric("Output File", output_path)

    # Video download
    if os.path.exists(output_path):
        with open(output_path, "rb") as vf:
            st.download_button(
                "⬇️ Download Annotated Video",
                data=vf,
                file_name="tracked_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    # Screenshots gallery
    if screenshots:
        st.markdown("#### 📸 Sample Frames")
        cols = st.columns(len(screenshots))
        for i, (col, img) in enumerate(zip(cols, screenshots)):
            with col:
                st.image(img, caption=f"Frame sample {i+1}", use_container_width=True)

elif run_button and not url:
    st.warning("⚠️ Please paste a YouTube URL in the sidebar first.")
else:
    # Landing state
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="status-card">
            <h3>Step 1</h3>
            <p>🎬 Paste YouTube URL</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="status-card">
            <h3>Step 2</h3>
            <p>⚙️ Configure & Detect Teams</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="status-card">
            <h3>Step 3</h3>
            <p>🚀 Run & Download Results</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 **Get started** by pasting a YouTube video URL in the sidebar and clicking **Run Pipeline**.")
