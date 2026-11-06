"""VisionCount Streamlit application."""
from pathlib import Path
import tempfile

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from src.visioncount.counter import VideoCounter
from src.visioncount.live import VisionCountLiveProcessor

st.set_page_config(page_title="VisionCount", page_icon="👥", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18); padding: 14px 16px; border-radius: 12px;}
div[data-testid="stTabs"] button {font-weight: 600;}
.vc-subtitle {color: #8b949e; margin-top: -10px; margin-bottom: 18px;}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([4, 1])
with left:
    st.title("👥 VisionCount")
    st.markdown('<div class="vc-subtitle">Human detection · Multi-object tracking · Directional people counting</div>', unsafe_allow_html=True)
with right:
    st.caption("VISION ANALYTICS")
    st.markdown("**YOLO + MOT**")

with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("Detection & tracking", expanded=True):
        weights = st.text_input("Detection model", "yolo11n.pt")
        confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
        image_size = st.select_slider("Inference resolution",
                                      options=[320, 416, 512, 640, 768, 960], value=640)
        device = st.selectbox("Processing device", ["cpu", "0"], index=0,
                              help="Choose GPU 0 only when CUDA-enabled PyTorch is installed.")
        tracker = st.selectbox("Tracking algorithm", ["bytetrack.yaml", "botsort.yaml"],
                               format_func=lambda value: "ByteTrack" if value.startswith("byte") else "BoT-SORT")

    with st.expander("Counting line", expanded=True):
        orientation_label = st.radio(
            "Line orientation", ["Horizontal", "Vertical"], horizontal=True,
            help="Horizontal separates top/bottom. Vertical separates left/right."
        )
        line_orientation = orientation_label.lower()
        if line_orientation == "horizontal":
            position_label = "Position from top (%)"
            direction_options = ["Top → Bottom", "Bottom → Top"]
            default_direction = "Top → Bottom"
        else:
            position_label = "Position from left (%)"
            direction_options = ["Left → Right", "Right → Left"]
            default_direction = "Left → Right"
        line_percent = st.slider(position_label, 5, 95, 50, 1)
        line_ratio = line_percent / 100.0
        in_direction_label = st.selectbox(
            "Count IN when moving", direction_options,
            index=direction_options.index(default_direction)
        )
        in_direction = (
            "negative_to_positive"
            if in_direction_label in ("Top → Bottom", "Left → Right")
            else "positive_to_negative"
        )
        cooldown = st.slider("Crossing debounce (seconds)", 0.0, 3.0, 0.75, 0.05)

    with st.expander("Visualization", expanded=False):
        trajectories = st.checkbox("Show movement trails", True)

source = st.radio("INPUT SOURCE", ["Upload video", "Live webcam"], horizontal=True)

def processor_factory():
    return VisionCountLiveProcessor(
        weights=weights, confidence=confidence, image_size=image_size,
        device=device, tracker=tracker, line_ratio=line_ratio,
        cooldown_seconds=cooldown, draw_trajectories=trajectories,
        line_orientation=line_orientation, in_direction=in_direction,
    )

if source == "Live webcam":
    st.subheader("Live camera")
    st.caption(f"{tracker.replace('.yaml', '')} · {orientation_label.lower()} line · IN: {in_direction_label}")
    st.info("Start the camera and allow browser camera access. The annotated stream appears below.")
    webrtc_streamer(
        key="visioncount-live-webcam-v2",
        video_processor_factory=processor_factory,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Browser webcam requires localhost or HTTPS. Stop the stream before changing tracking or line settings.")

else:
    st.subheader("Video analysis")
    st.caption("Upload a clip, configure the counting boundary in the sidebar, then run detection and tracking.")
    uploaded = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV."
    )
    if uploaded:
        with st.container(border=True):
            file_col, settings_col = st.columns([3, 1])
            with file_col:
                st.markdown(f"**{uploaded.name}**")
                st.caption(f"{uploaded.size / (1024 * 1024):.1f} MB")
            with settings_col:
                st.markdown(f"**{orientation_label} line**")
                st.caption(f"{line_percent}% · IN: {in_direction_label}")
            with st.expander("Preview original video"):
                st.video(uploaded)

        run_col, detail_col = st.columns([1, 3], vertical_alignment="center")
        with run_col:
            run_analysis = st.button("▶  Run analysis", type="primary", use_container_width=True)
        with detail_col:
            st.caption(f"Model: {weights} · Tracker: {tracker.replace('.yaml', '')} · Device: {device}")

        if run_analysis:
            progress = st.progress(0.0, text="Initializing pipeline...")
            status = st.empty()
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    input_path = Path(temp_dir) / Path(uploaded.name).name
                    output_path = Path(temp_dir) / "visioncount_annotated.mp4"
                    input_path.write_bytes(uploaded.getvalue())
                    counter = VideoCounter(
                        weights=weights, confidence=confidence, image_size=image_size,
                        device=device, tracker=tracker, line_ratio=line_ratio,
                        cooldown_seconds=cooldown, draw_trajectories=trajectories,
                        line_orientation=line_orientation, in_direction=in_direction,
                    )

                    def update_progress(value):
                        progress.progress(value, text=f"Processing frames · {value:.0%}")

                    status.info("Loading model and processing video. First run may download model weights.")
                    metrics = counter.process(input_path, output_path, update_progress)
                    video_bytes = output_path.read_bytes()

                status.empty()
                progress.progress(1.0, text="Analysis complete")
                st.success("Video analysis completed.")
                st.markdown("### Analysis summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("People IN", metrics["entered"])
                m2.metric("People OUT", metrics["exited"])
                m3.metric("Occupancy estimate", metrics["occupancy"])
                m4.metric("Processing FPS", f'{metrics["processing_fps"]:.1f}')

                video_tab, chart_tab, events_tab = st.tabs(
                    ["▶  Processed video", "📈  Count timeline", "☷  Crossing events"]
                )
                with video_tab:
                    st.markdown("**Annotated playback**")
                    st.video(video_bytes)
                    st.caption("Bounding boxes, track IDs, centroids, trajectories and the configured counting line are embedded in the video.")
                with chart_tab:
                    st.markdown("**Cumulative people crossing over video time**")
                    timeline = pd.DataFrame(metrics["timeline"])
                    chart_data = timeline.melt(
                        id_vars=["video_time_seconds"],
                        value_vars=["IN", "OUT"],
                        var_name="Direction",
                        value_name="People",
                    )
                    chart = (
                        alt.Chart(chart_data)
                        .mark_line(point=alt.OverlayMarkDef(filled=True, size=55), strokeWidth=3)
                        .encode(
                            x=alt.X("video_time_seconds:Q", title="Video timeline (seconds)"),
                            y=alt.Y("People:Q", title="Number of people", scale=alt.Scale(zero=True)),
                            color=alt.Color(
                                "Direction:N", title="Crossing direction",
                                scale=alt.Scale(domain=["IN", "OUT"], range=["#16a34a", "#e05a4f"])
                            ),
                            tooltip=[
                                alt.Tooltip("video_time_seconds:Q", title="Video time", format=".2f"),
                                alt.Tooltip("Direction:N"),
                                alt.Tooltip("People:Q", title="Cumulative count"),
                            ],
                        )
                        .properties(height=420)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("IN and OUT are cumulative crossing totals. Occupancy is estimated separately as IN minus OUT.")
                with events_tab:
                    events = pd.DataFrame(
                        metrics["events"],
                        columns=["timestamp_seconds", "track_id", "direction", "center_x", "center_y"],
                    )
                    st.dataframe(events, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download crossing events CSV",
                        events.to_csv(index=False).encode("utf-8"),
                        "visioncount_events.csv", "text/csv"
                    )
                st.caption(
                    f'Processed {metrics["frames"]:,} frames · '
                    f'{metrics["duration_seconds"]:.1f}s video duration · '
                    f'Peak estimated occupancy: {metrics["peak_occupancy"]}.'
                )
                st.warning("Counting accuracy depends on tracker association and camera placement. A crossing is counted only when a tracked person's centroid changes sides.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)
    else:
        st.info("Choose a video above to begin. Configure line orientation, position and IN direction in the sidebar.")
