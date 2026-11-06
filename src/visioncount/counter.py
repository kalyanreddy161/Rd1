"""OpenCV video pipeline for detection, tracking, and counting."""
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np

from .analytics import LineCrossingAnalytics
from .detector import PersonDetector


class VideoCounter:
    def __init__(self, weights="yolo11n.pt", confidence=0.35, image_size=640,
                 device="cpu", tracker="bytetrack.yaml", line_ratio=0.5,
                 cooldown_seconds=0.75, draw_trajectories=True):
        self.detector = PersonDetector(weights, confidence, image_size, device, tracker)
        self.line_ratio = float(line_ratio)
        self.cooldown_seconds = float(cooldown_seconds)
        self.draw_trajectories = bool(draw_trajectories)

    def process(self, input_path, output_path, progress_callback=None):
        output_path = Path(output_path)
        raw_output_path = output_path.with_name(output_path.stem + "_opencv.mp4")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video has invalid frame dimensions.")

        line_y = int(height * self.line_ratio)
        analytics = LineCrossingAnalytics(line_y, self.cooldown_seconds)
        writer = cv2.VideoWriter(
            str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create intermediate video. Check output directory/codecs.")

        trajectories = {}
        timeline = [{"video_time_seconds": 0.0, "IN": 0, "OUT": 0, "People inside (estimate)": 0}]
        next_sample_time = 1.0
        frame_index = 0
        started = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                timestamp = frame_index / fps
                results = self.detector.track(frame)
                previous_event_count = len(analytics.events)
                annotated = self._annotate(
                    frame, results[0], analytics, trajectories,
                    line_y, frame_index / max(time.perf_counter() - started, 1e-6), timestamp
                )
                writer.write(annotated)

                # Keep a compact timeline: one sample per second plus every crossing.
                if timestamp >= next_sample_time or len(analytics.events) > previous_event_count:
                    timeline.append({
                        "video_time_seconds": round(timestamp, 3),
                        "IN": analytics.entered,
                        "OUT": analytics.exited,
                        "People inside (estimate)": analytics.occupancy,
                    })
                    while next_sample_time <= timestamp:
                        next_sample_time += 1.0

                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_index / total_frames, 1.0))
        finally:
            cap.release()
            writer.release()

        if frame_index == 0:
            raise RuntimeError("The selected video contains no readable frames.")

        final_timestamp = frame_index / fps
        if timeline[-1]["video_time_seconds"] != round(final_timestamp, 3):
            timeline.append({
                "video_time_seconds": round(final_timestamp, 3),
                "IN": analytics.entered,
                "OUT": analytics.exited,
                "People inside (estimate)": analytics.occupancy,
            })

        # Convert OpenCV's intermediate MPEG-4 output to H.264 for reliable
        # in-browser playback in Streamlit/Chrome.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-y", "-i", str(raw_output_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"Could not encode browser-compatible H.264 video: {detail[-1500:]}") from exc
        finally:
            raw_output_path.unlink(missing_ok=True)

        elapsed = max(time.perf_counter() - started, 1e-6)
        return {
            "frames": frame_index,
            "entered": analytics.entered,
            "exited": analytics.exited,
            "occupancy": analytics.occupancy,
            "peak_occupancy": analytics.peak_occupancy,
            "processing_fps": frame_index / elapsed,
            "events": analytics.events_as_dicts(),
            "timeline": timeline,
            "duration_seconds": final_timestamp,
        }

    def _annotate(self, frame, result, analytics, trajectories,
                  line_y, processing_fps, timestamp_seconds):
        height, width = frame.shape[:2]
        annotated = frame.copy()
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 220, 255), 2)
        cv2.putText(annotated, "COUNTING LINE", (12, max(24, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            coordinates = boxes.xyxy.cpu().numpy().astype(int)
            ids = (boxes.id.cpu().numpy().astype(int).tolist()
                   if boxes.id is not None else [None] * len(coordinates))
            confidences = (boxes.conf.cpu().numpy().tolist()
                           if boxes.conf is not None else [None] * len(coordinates))

            for box, track_id, confidence in zip(coordinates, ids, confidences):
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = f"ID {track_id}" if track_id is not None else "Person"
                if confidence is not None:
                    label += f" {confidence:.2f}"

                if track_id is not None:
                    event = analytics.update(track_id, cx, cy, timestamp_seconds)
                    points = trajectories.setdefault(int(track_id), [])
                    points.append((cx, cy))
                    if len(points) > 40:
                        del points[:-40]
                    if event:
                        cv2.putText(annotated, event.direction, (x1, min(height - 8, y2 + 22)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)

        if self.draw_trajectories:
            for points in trajectories.values():
                if len(points) > 1:
                    cv2.polylines(annotated, [np.array(points, dtype=np.int32)],
                                  False, (255, 120, 0), 2)

        current_people = len(boxes) if boxes is not None else 0
        cv2.rectangle(annotated, (8, 8), (350, 130), (25, 25, 25), -1)
        cv2.putText(annotated, f"Detected: {current_people}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        cv2.putText(annotated, f"IN: {analytics.entered}  OUT: {analytics.exited}",
                    (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"Occupancy estimate: {analytics.occupancy}",
                    (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"Processing: {processing_fps:.1f} FPS",
                    (18, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return annotated
