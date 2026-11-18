import unittest

from scripts.build_bundles_s2 import _year_overlap
from scripts.build_kb_index import extract_fields, parse_time_interval


class S2TemporalSemanticsTests(unittest.TestCase):
    def test_supported_label_value_formats(self):
        text = "URI | https://example.org/x\nDefinition | An exact concept definition.\n| Country | Belgium |\n**Data holding centre:** Marine Centre"
        fields = extract_fields(text)
        self.assertEqual(fields["uri"], "https://example.org/x")
        self.assertEqual(fields["definition"], "An exact concept definition.")
        self.assertEqual(fields["country"], "Belgium")
        self.assertEqual(fields["data holding centre"], "Marine Centre")

    def test_open_end_intervals_overlap_later_ranges(self):
        interval = parse_time_interval("From 1983 onwards")
        self.assertEqual(interval, {"start_year": 1983, "end_year": None, "open_end": True})
        self.assertTrue(_year_overlap(interval, 1989, 1992))

    def test_ongoing_without_a_start_year_is_not_assumed_to_cover_past_years(self):
        interval = parse_time_interval("Ongoing")
        self.assertFalse(_year_overlap(interval, 1989, 1992))

    def test_short_year_range_is_expanded_with_century(self):
        interval = parse_time_interval("1971-75")
        self.assertTrue(_year_overlap(interval, 1972, 1974))
        self.assertFalse(_year_overlap(interval, 1989, 1992))


if __name__ == "__main__":
    unittest.main()
