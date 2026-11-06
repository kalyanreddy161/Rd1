import unittest

from src.visioncount.analytics import LineCrossingAnalytics


class TestLineCrossingAnalytics(unittest.TestCase):
    def test_horizontal_top_to_bottom_is_in(self):
        analytics = LineCrossingAnalytics(line_y=100, cooldown_seconds=0)
        analytics.update(1, 20, 80, 0.0)
        event = analytics.update(1, 20, 120, 1.0)
        self.assertEqual(event.direction, "IN")
        self.assertEqual(analytics.entered, 1)

    def test_horizontal_bottom_to_top_is_out(self):
        analytics = LineCrossingAnalytics(line_y=100, cooldown_seconds=0)
        analytics.update(1, 20, 120, 0.0)
        event = analytics.update(1, 20, 80, 1.0)
        self.assertEqual(event.direction, "OUT")
        self.assertEqual(analytics.exited, 1)

    def test_vertical_left_to_right_is_in(self):
        analytics = LineCrossingAnalytics(
            cooldown_seconds=0, orientation="vertical", line_position=100
        )
        analytics.update(7, 80, 25, 0.0)
        event = analytics.update(7, 120, 25, 1.0)
        self.assertEqual(event.direction, "IN")
        self.assertEqual(analytics.entered, 1)

    def test_vertical_right_to_left_can_be_in(self):
        analytics = LineCrossingAnalytics(
            cooldown_seconds=0, orientation="vertical", line_position=100,
            in_direction="positive_to_negative"
        )
        analytics.update(9, 120, 25, 0.0)
        event = analytics.update(9, 80, 25, 1.0)
        self.assertEqual(event.direction, "IN")

    def test_no_event_without_side_change(self):
        analytics = LineCrossingAnalytics(line_y=100)
        analytics.update(1, 20, 80, 0.0)
        self.assertIsNone(analytics.update(1, 25, 90, 1.0))
        self.assertEqual(len(analytics.events), 0)

    def test_separate_track_ids(self):
        analytics = LineCrossingAnalytics(line_y=100, cooldown_seconds=0)
        analytics.update(1, 20, 80, 0.0)
        analytics.update(2, 40, 80, 0.0)
        analytics.update(1, 20, 120, 1.0)
        analytics.update(2, 40, 120, 1.0)
        self.assertEqual(analytics.entered, 2)


if __name__ == "__main__":
    unittest.main()
