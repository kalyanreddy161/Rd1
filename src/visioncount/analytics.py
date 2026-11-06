"""Directional crossing analytics for tracked people."""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import time


@dataclass
class CrossingEvent:
    timestamp_seconds: float
    track_id: int
    direction: str
    center_x: int
    center_y: int


class LineCrossingAnalytics:
    """Count centroid crossings of a horizontal or vertical virtual line.

    Negative-to-positive movement is IN by default: top-to-bottom for a
    horizontal line and left-to-right for a vertical line.
    """

    def __init__(self, line_y: Optional[int] = None, cooldown_seconds: float = 0.75,
                 orientation: str = "horizontal", line_position: Optional[int] = None,
                 in_direction: str = "negative_to_positive"):
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        if in_direction not in {"negative_to_positive", "positive_to_negative"}:
            raise ValueError("invalid in_direction")
        if line_position is None:
            if line_y is None:
                raise ValueError("line_position (or legacy line_y) is required")
            line_position = line_y
        self.orientation = orientation
        self.line_position = int(line_position)
        self.line_y = self.line_position if orientation == "horizontal" else None
        self.cooldown_seconds = float(cooldown_seconds)
        self.in_direction = in_direction
        self.previous_side: Dict[int, int] = {}
        self.last_crossing: Dict[int, float] = {}
        self.entered = 0
        self.exited = 0
        self.events = []

    def update(self, track_id: int, center_x: int, center_y: int,
               timestamp_seconds: Optional[float] = None):
        now = time.monotonic() if timestamp_seconds is None else float(timestamp_seconds)
        coordinate = center_y if self.orientation == "horizontal" else center_x
        side = -1 if coordinate < self.line_position else 1
        previous = self.previous_side.get(track_id)
        self.previous_side[track_id] = side
        if previous is None or previous == side:
            return None
        if now - self.last_crossing.get(track_id, float("-inf")) < self.cooldown_seconds:
            return None

        negative_to_positive = previous == -1 and side == 1
        is_in = (negative_to_positive if self.in_direction == "negative_to_positive"
                 else not negative_to_positive)
        direction = "IN" if is_in else "OUT"
        self.last_crossing[track_id] = now
        if is_in:
            self.entered += 1
        else:
            self.exited += 1
        event = CrossingEvent(now, int(track_id), direction, int(center_x), int(center_y))
        self.events.append(event)
        return event

    @property
    def occupancy(self) -> int:
        return max(0, self.entered - self.exited)

    @property
    def peak_occupancy(self) -> int:
        occupancy = 0
        peak = 0
        for event in self.events:
            occupancy += 1 if event.direction == "IN" else -1
            occupancy = max(0, occupancy)
            peak = max(peak, occupancy)
        return peak

    def events_as_dicts(self):
        return [asdict(event) for event in self.events]
