"""Streamlit interface for VisionCount."""
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.visioncount.counter import VideoCounter

st.set_page_config(page_title="VisionCount", page_icon="👥", layout="wide")
st.title("👥 VisionCount")
st.caption("Real-time human detection, multi-object tracking, and directional people counting.")

with st.sidebar:
    st.header("Detection settings")
    weights = st.text_input("YOLO weights", "yolo11n.pt",
                             help="Default model is downloaded automatically on first run.")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    device = st.selectbox("Device", ["cpu", "0"], index=0,
                          help="Choose 0 only after confirming CUDA-enabled PyTorch is installed.")
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    line_ratio = st.slider("Counting line vertical position", 0.10, 0.90, 0.50, 0.05)
    cooldown = st.slider("Crossing cooldown (seconds)", 0.0, 3.0, 0.75, 0.05)
    trajectories = st.checkbox("Draw recent trajectories", True)

st.info("Upload a video to begin. Webcam and RTSP live-stream support are planned for a later iteration.")
uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    st.video(uploaded)
    if st.button("Run people counting", type="primary"):
        progress = st.progress(0.0, text="Preparing video...")
        status = st.empty()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / uploaded.name
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

                status.info("Loading detector and processing frames. The first run may download model weights.")
                metrics = counter.process(input_path, output_path, update_progress)
                video_bytes = output_path.read_bytes()
                heatmap_bytes = Path(metrics["heatmap_path"]).read_bytes()

                st.success("Video processing completed.")
                a, b, c, d = st.columns(4)
                a.metric("People entered", metrics["entered"])
                b.metric("People exited", metrics["exited"])
                c.metric("Occupancy estimate", metrics["occupancy"])
                d.metric("Processing FPS", f'{metrics["processing_fps"]:.1f}')

                st.subheader("Annotated output")
                st.video(video_bytes)
                st.download_button("Download annotated video", video_bytes,
                                   "visioncount_annotated.mp4", "video/mp4")
                st.subheader("Movement heatmap")
                st.image(heatmap_bytes, use_container_width=True)
                events = pd.DataFrame(metrics["events"],
                                      columns=["timestamp_seconds", "track_id", "direction", "center_x", "center_y"])
                st.subheader("Crossing events")
                st.dataframe(events, use_container_width=True)
                st.download_button("Download crossing events (CSV)",
                                   events.to_csv(index=False).encode("utf-8"),
                                   "visioncount_events.csv", "text/csv")
                st.caption(f'Processed {metrics["frames"]} frames. Peak derived occupancy: {metrics["peak_occupancy"]}.')
                st.warning("Counts depend on correct track association and line-crossing detection. Review the output for missed or duplicate crossings.")
                progress.progress(1.0, text="Done")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.exception(exc)
else:
    st.markdown("### Workflow")
    st.markdown("1. Upload a video.  2. Select the detector and tracker.  3. Set the virtual counting line.  4. Run analysis and export results.")
