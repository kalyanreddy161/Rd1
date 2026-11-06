"""Live webcam frame processor for Streamlit-WebRTC."""
import time

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from .analytics import LineCrossingAnalytics
from .detector import PersonDetector


class VisionCountLiveProcessor(VideoProcessorBase):
    """Run persistent YOLO tracking on browser webcam frames."""

    def __init__(self, weights="yolo11n.pt", confidence=0.35, image_size=640,
                 device="cpu", tracker="bytetrack.yaml", line_ratio=0.5,
                 cooldown_seconds=0.75, draw_trajectories=True):
        self.detector = PersonDetector(weights, confidence, image_size, device, tracker)
        self.line_ratio = float(line_ratio)
        self.cooldown_seconds = float(cooldown_seconds)
        self.draw_trajectories = bool(draw_trajectories)
        self.analytics = None
        self.trajectories = {}
        self.frame_count = 0
        self.started = time.perf_counter()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        if self.analytics is None:
            self.analytics = LineCrossingAnalytics(
                int(height * self.line_ratio), self.cooldown_seconds
            )

        result = self.detector.track(image)[0]
        self.frame_count += 1
        fps = self.frame_count / max(time.perf_counter() - self.started, 1e-6)
        output = image.copy()
        line_y = self.analytics.line_y
        cv2.line(output, (0, line_y), (width, line_y), (0, 220, 255), 2)
        cv2.putText(output, "COUNTING LINE", (12, max(24, line_y - 10)),
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
                    event = self.analytics.update(track_id, cx, cy)
                    points = self.trajectories.setdefault(int(track_id), [])
                    points.append((cx, cy))
                    if len(points) > 40:
                        del points[:-40]
                    if event:
                        cv2.putText(output, event.direction, (x1, min(height - 8, y2 + 22)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
                cv2.rectangle(output, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(output, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)

        if self.draw_trajectories:
            for points in self.trajectories.values():
                if len(points) > 1:
                    cv2.polylines(output, [np.array(points, dtype=np.int32)],
                                  False, (255, 120, 0), 2)

        detected = len(boxes) if boxes is not None else 0
        cv2.rectangle(output, (8, 8), (350, 130), (25, 25, 25), -1)
        cv2.putText(output, f"Detected: {detected}", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        cv2.putText(output, f"IN: {self.analytics.entered}  OUT: {self.analytics.exited}",
                    (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(output, f"Occupancy estimate: {self.analytics.occupancy}",
                    (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(output, f"Processing: {fps:.1f} FPS", (18, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return av.VideoFrame.from_ndarray(output, format="bgr24")
