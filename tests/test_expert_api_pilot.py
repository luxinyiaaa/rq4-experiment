import unittest

from scripts.expert_api_pilot import extract_urls


class ExpertApiPilotUrlTests(unittest.TestCase):
    def test_extracts_plain_markdown_and_fullwidth_bracket_urls(self):
        answer = (
            "Plain https://example.org/plain. "
            "Markdown [source](https://example.org/markdown). "
            "Bracketed 【https://example.org/bracket】"
        )
        self.assertEqual(
            extract_urls(answer),
            [
                "https://example.org/plain",
                "https://example.org/markdown",
                "https://example.org/bracket",
            ],
        )

    def test_strips_closing_punctuation_and_ignores_non_url_citation_marker(self):
        answer = "Cite 【0†a citation marker】 and https://example.org/path),.;"
        self.assertEqual(extract_urls(answer), ["https://example.org/path"])

    def test_deduplicates_markdown_and_plain_occurrences(self):
        self.assertEqual(
            extract_urls("[link](https://example.org/x) https://example.org/x."),
            ["https://example.org/x"],
        )


if __name__ == "__main__":
    unittest.main()
