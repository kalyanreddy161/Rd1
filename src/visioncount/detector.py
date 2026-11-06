"""YOLO model construction and tracking inference."""
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, weights="yolo11n.pt", confidence=0.35, image_size=640,
                 device="cpu", tracker="bytetrack.yaml"):
        self.model = YOLO(weights)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device
        self.tracker = tracker

    def track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],  # COCO class 0 = person
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
