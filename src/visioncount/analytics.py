"""Directional line-crossing analytics for tracked people."""
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
    """Count a track once when its centroid changes side of a horizontal line.

    The line is interpreted in image coordinates. Moving from above to below
    is an entry; moving from below to above is an exit. This convention can be
    changed in the dashboard depending on camera placement.
    """

    def __init__(self, line_y: int, cooldown_seconds: float = 0.75):
        self.line_y = int(line_y)
        self.cooldown_seconds = float(cooldown_seconds)
        self.previous_side: Dict[int, int] = {}
        self.last_crossing: Dict[int, float] = {}
        self.entered = 0
        self.exited = 0
        self.events = []

    def update(self, track_id: int, center_x: int, center_y: int,
               timestamp_seconds: Optional[float] = None):
        now = time.monotonic() if timestamp_seconds is None else float(timestamp_seconds)
        side = -1 if center_y < self.line_y else 1
        previous = self.previous_side.get(track_id)
        self.previous_side[track_id] = side

        if previous is None or previous == side:
            return None

        if now - self.last_crossing.get(track_id, float("-inf")) < self.cooldown_seconds:
            return None

        # Above -> below is IN; below -> above is OUT.
        direction = "IN" if previous == -1 and side == 1 else "OUT"
        self.last_crossing[track_id] = now
        if direction == "IN":
            self.entered += 1
        else:
            self.exited += 1

        event = CrossingEvent(now, int(track_id), direction, int(center_x), int(center_y))
        self.events.append(event)
        return event

    @property
    def occupancy(self) -> int:
        # This is a derived estimate. A video may begin mid-session or miss
        # crossings, so it is not guaranteed to represent physical occupancy.
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
