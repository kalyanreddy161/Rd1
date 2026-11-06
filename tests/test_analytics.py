import unittest

from src.visioncount.analytics import LineCrossingAnalytics


class TestLineCrossingAnalytics(unittest.TestCase):
    def test_counts_in_and_out(self):
        analytics = LineCrossingAnalytics(line_y=100, cooldown_seconds=0)
        analytics.update(1, 20, 80, 0.0)
        event_in = analytics.update(1, 20, 120, 1.0)
        self.assertEqual(event_in.direction, "IN")
        self.assertEqual(analytics.entered, 1)
        event_out = analytics.update(1, 20, 80, 2.0)
        self.assertEqual(event_out.direction, "OUT")
        self.assertEqual(analytics.exited, 1)
        self.assertEqual(analytics.occupancy, 0)

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
