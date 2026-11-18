"""Mock-judge tests for window screening and same-category replacement."""

import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts.sample_windows import QUOTAS
from scripts.screen_windows import (
    _duplicate_reason,
    judge_window,
    replace_specific_windows,
    screen_windows,
)


class WindowScreeningTests(unittest.TestCase):
    def test_judge_window_uses_json_judge_and_validates_decision(self):
        calls = []

        def mock_call(role, messages, **kwargs):
            calls.append((role, messages, kwargs))
            return {"json": {"decision": "suitable", "reason": "It states a complete fact."}}

        result = judge_window({"page_type": "about_science", "url": "https://example.org/x", "text": "Fact."}, mock_call)

        self.assertEqual(result["decision"], "suitable")
        role, messages, kwargs = calls[0]
        self.assertEqual(role, "judge")
        self.assertTrue(kwargs["json_mode"])
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertIn("screening a short evidence window", messages[0]["content"])

    def test_unsuitable_window_is_replaced_by_suitable_same_type_candidate(self):
        windows = []
        for page_type, count in QUOTAS.items():
            for index in range(count):
                domain = f"{page_type}-{index // 5}.example"
                windows.append({
                    "window_id": f"W{len(windows) + 1:03d}",
                    "url": f"https://{domain}/{index}",
                    "domain": domain,
                    "page_type": page_type,
                    "chunk_ids": [str(index)],
                    "text": "original text" if not windows else f"text {len(windows)}",
                })
        candidates = {page_type: [] for page_type in QUOTAS}
        candidates["dataset_record"].append({
            "url": "https://replacement.example/record",
            "domain": "replacement.example",
            "page_type": "dataset_record",
            "chunk_ids": ["replacement-chunk"],
            "text": "replacement text",
            "length": 1200,
            "length_rule": "900–1,500 chars",
        })

        def mock_call(role, messages, **kwargs):
            text = messages[1]["content"]
            decision = "unsuitable" if "original text" in text else "suitable"
            reason = "It is a fragment." if decision == "unsuitable" else "It states a complete fact."
            return {"json": {"decision": decision, "reason": reason}}

        with redirect_stdout(StringIO()):
            screened, replacements, unfilled = screen_windows(windows, candidates, llm_call=mock_call, final_counts=QUOTAS)

        self.assertEqual(len(screened), 50)
        self.assertEqual(screened[-1]["url"], "https://replacement.example/record")
        self.assertEqual(screened[-1]["page_type"], "dataset_record")
        self.assertEqual(len(replacements), 1)
        self.assertEqual(unfilled, [])
        self.assertEqual(replacements[0]["replacement_reason"], "It is a fragment.")
        self.assertTrue(replacements[0]["selected_as_replacement"])

    def test_dataset_shortfall_can_be_filled_from_about_science(self):
        windows = []
        for page_type, count in QUOTAS.items():
            for index in range(count):
                domain = f"{page_type}-{index // 5}.example"
                windows.append({
                    "window_id": f"W{len(windows) + 1:03d}",
                    "url": f"https://{domain}/{index}",
                    "domain": domain,
                    "page_type": page_type,
                    "chunk_ids": [str(index)],
                    "text": "bad" if page_type == "dataset_record" and index < 2 else f"text {len(windows)}",
                })
        candidates = {kind: [] for kind in QUOTAS}
        for i in range(2):
            candidates["about_science"].append({
                "url": f"https://replacement-{i}.example/page",
                "domain": f"replacement-{i}.example",
                "page_type": "about_science",
                "chunk_ids": [f"replacement-{i}"],
                "text": f"replacement {i}",
                "length": 1200,
                "length_rule": "900–1,500 chars",
            })

        def mock_call(role, messages, **kwargs):
            decision = "unsuitable" if "bad" in messages[1]["content"] else "suitable"
            return {"json": {
                "decision": decision,
                "reason": "It is a fragment." if decision == "unsuitable" else "It states a complete fact.",
            }}

        targets = {"dataset_record": 23, "about_science": 22, "vocabulary": 5}
        with redirect_stdout(StringIO()):
            screened, replacements, unfilled = screen_windows(
                windows, candidates, llm_call=mock_call, final_counts=targets
            )

        self.assertEqual(len(screened), 50)
        self.assertEqual({k: sum(item["page_type"] == k for item in screened) for k in targets}, targets)
        self.assertEqual(unfilled, [])
        cross_type = [row for row in replacements if row["candidate_page_type"] == "about_science"]
        self.assertEqual(len(cross_type), 2)
        self.assertTrue(all(row["selected_as_replacement"] for row in cross_type))

    def test_targeted_replacement_skips_same_dataset_and_preserves_other_windows(self):
        windows = []
        for page_type, count in QUOTAS.items():
            for index in range(count):
                text = f"Unique factual passage {len(windows)} with a value {index}."
                if len(windows) < 2:
                    text = "# Shared dataset title\nDataset facts and measurements are described here."
                windows.append({
                    "window_id": f"W{len(windows) + 1:03d}",
                    "url": f"https://{page_type}-{index}.example/{index}",
                    "domain": f"{page_type}-{index}.example",
                    "page_type": page_type,
                    "chunk_ids": [str(index)],
                    "text": text,
                })
        original_others = [item.copy() for item in windows[1:]]
        candidates = {kind: [] for kind in QUOTAS}
        candidates["dataset_record"] = [
            {
                "url": "https://candidate.example/same",
                "domain": "candidate.example",
                "page_type": "dataset_record",
                "chunk_ids": ["same"],
                "text": "# Shared dataset title\nDifferent metadata but the same dataset.",
                "length": 1200,
                "length_rule": "900–1,500 chars",
            },
            {
                "url": "https://candidate.example/other",
                "domain": "candidate.example",
                "page_type": "dataset_record",
                "chunk_ids": ["other"],
                "text": "# Other dataset title\nAn independent data record reports a complete measurement fact.",
                "length": 1200,
                "length_rule": "900–1,500 chars",
            },
        ]

        calls = []

        def mock_call(role, messages, **kwargs):
            calls.append(role)
            return {"json": {"decision": "suitable", "reason": "It states a complete fact."}}

        with redirect_stdout(StringIO()):
            screened, rows = replace_specific_windows(
                windows,
                ["W001"],
                candidates,
                {kind: [] for kind in QUOTAS},
                llm_call=mock_call,
            )

        self.assertEqual(len(screened), 50)
        self.assertEqual(screened[1:], original_others)
        self.assertEqual([row["candidate_decision"] for row in rows], ["unsuitable", "suitable"])
        self.assertIn("Same dataset title", rows[0]["candidate_reason"])
        self.assertEqual(calls, ["judge"])

    def test_similarity_above_threshold_is_unsuitable(self):
        base = {"window_id": "W1", "page_type": "about_science", "text": "A" * 1000}
        candidate = {"window_id": "W2", "page_type": "about_science", "text": "A" * 950 + "B" * 50}
        self.assertIn("Text similarity", _duplicate_reason(candidate, [base]))


if __name__ == "__main__":
    unittest.main()
