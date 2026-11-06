"""Streamlit UI: upload a video or use a live browser webcam."""
from pathlib import Path
import tempfile

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from src.visioncount.counter import VideoCounter
from src.visioncount.live import VisionCountLiveProcessor

st.set_page_config(page_title="VisionCount", page_icon="👥", layout="wide")
st.title("👥 VisionCount")
st.caption("Human detection, multi-object tracking, and people counting.")

with st.sidebar:
    st.header("Detection settings")
    weights = st.text_input("YOLO weights", "yolo11n.pt")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    image_size = st.select_slider("Inference image size",
                                  options=[320, 416, 512, 640, 768, 960], value=640)
    device = st.selectbox("Device", ["cpu", "0"], index=0,
                          help="Select 0 only when CUDA-enabled PyTorch is installed.")
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    line_ratio = st.slider("Counting line vertical position", 0.10, 0.90, 0.50, 0.05)
    cooldown = st.slider("Crossing cooldown (seconds)", 0.0, 3.0, 0.75, 0.05)
    trajectories = st.checkbox("Draw recent trajectories", True)

source = st.radio("Video source", ["Upload video", "Webcam (live)"], horizontal=True)

if source == "Webcam (live)":
    st.subheader("Live webcam detection")
    st.write("Press **START** in the webcam panel and allow browser camera access. The annotated stream appears below.")
    st.warning("Webcam mode uses your browser camera through WebRTC. It works on localhost and HTTPS deployments; plain HTTP remote hosting generally cannot access the camera.")
    processor_factory = lambda: VisionCountLiveProcessor(
        weights=weights,
        confidence=confidence,
        image_size=image_size,
        device=device,
        tracker=tracker,
        line_ratio=line_ratio,
        cooldown_seconds=cooldown,
        draw_trajectories=trajectories,
    )
    webrtc_streamer(
        key="visioncount-live-webcam",
        video_processor_factory=processor_factory,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Live IDs are tracker IDs, not guaranteed permanent identities. Stop the stream before changing tracker settings.")

else:
    st.subheader("Upload and analyze a video")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        st.video(uploaded)
        if st.button("Run detection and tracking", type="primary"):
            progress = st.progress(0.0, text="Preparing video...")
            status = st.empty()
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    input_path = Path(temp_dir) / Path(uploaded.name).name
                    output_path = Path(temp_dir) / "visioncount_annotated.mp4"
                    input_path.write_bytes(uploaded.getvalue())
                    counter = VideoCounter(
                        weights=weights,
                        confidence=confidence,
                        image_size=image_size,
                        device=device,
                        tracker=tracker,
                        line_ratio=line_ratio,
                        cooldown_seconds=cooldown,
                        draw_trajectories=trajectories,
                    )

                    def update_progress(value):
                        progress.progress(value, text=f"Processing video: {value:.0%}")

                    status.info("Loading YOLO and processing frames. First run may download model weights.")
                    metrics = counter.process(input_path, output_path, update_progress)
                    video_bytes = output_path.read_bytes()

                    st.success("Processing completed.")
                    st.subheader("Processed video — play it here")
                    st.video(video_bytes)

                    a, b, c, d = st.columns(4)
                    a.metric("People entered", metrics["entered"])
                    b.metric("People exited", metrics["exited"])
                    c.metric("Occupancy estimate", metrics["occupancy"])
                    d.metric("Processing FPS", f'{metrics["processing_fps"]:.1f}')

                    st.subheader("People count over video timeline")
                    timeline = pd.DataFrame(metrics["timeline"])
                    chart_data = timeline.melt(
                        id_vars=["video_time_seconds"],
                        value_vars=["IN", "OUT"],
                        var_name="Direction",
                        value_name="Number of people",
                    )
                    count_chart = (
                        alt.Chart(chart_data)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("video_time_seconds:Q", title="Video timeline (seconds)"),
                            y=alt.Y("Number of people:Q", title="Number of people", scale=alt.Scale(zero=True)),
                            color=alt.Color("Direction:N", title="Crossing direction"),
                            tooltip=[
                                alt.Tooltip("video_time_seconds:Q", title="Video time", format=".1f"),
                                alt.Tooltip("Direction:N"),
                                alt.Tooltip("Number of people:Q"),
                            ],
                        )
                        .properties(height=380)
                        .interactive()
                    )
                    st.altair_chart(count_chart, use_container_width=True)
                    st.caption(
                        "IN and OUT are cumulative crossing counts. The graph records "
                        "one-second samples and additional points when a crossing occurs."
                    )

                    events = pd.DataFrame(
                        metrics["events"],
                        columns=["timestamp_seconds", "track_id", "direction", "center_x", "center_y"],
                    )
                    st.subheader("Crossing events")
                    st.dataframe(events, use_container_width=True)
                    st.download_button("Download crossing events (CSV)",
                                       events.to_csv(index=False).encode("utf-8"),
                                       "visioncount_events.csv", "text/csv")
                    st.caption(
                        f'Processed {metrics["frames"]} frames over '
                        f'{metrics["duration_seconds"]:.1f} seconds. '
                        f'Peak derived occupancy: {metrics["peak_occupancy"]}.'
                    )
                    st.warning(
                        "Counts depend on correct tracking and line-crossing detection. "
                        "Review the annotated video for missed or duplicate crossings."
                    )
                    progress.progress(1.0, text="Done")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)
    else:
        st.info("Upload an MP4, AVI, MOV, or MKV file. The processed video will play in this app with bounding boxes, IDs, trajectories, and counters.")
