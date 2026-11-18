"""Mock tests for the S2 pilot generation and evidence-checking workflow."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from scripts.check_questions_s2 import multi_page_check, quote_verbatim_check, set_completeness_check, _rules, _build_bm25_index, _quote_support_check
from scripts.build_bundles_s2 import _comparison_value_in_any_title, _note, _parse_comparison_value, _shared_comparison_relation
from scripts.generate_questions_s2 import generate_one, generate_pilot, response_issues, select_pilot


def sample_bundle(kind="multi_hop"):
    pages = [
        {"evidence_id": "E1", "url": "https://example.org/a", "title": "Record Alpha", "text": "A concrete measurement value is 12 units.", "chunk_ids": ["c1"]},
        {"evidence_id": "E2", "url": "https://example.org/b", "title": "Center Beta", "text": "The center maintains the related instrument record.", "chunk_ids": ["c2"]},
    ]
    return {"bundle_id": "S2-MULTI_HOP-001", "type": kind, "user_task": "dataset_details", "pages": pages,
            "group_note": "", "s1_note": "", "difficulty": {"page_count": 2}}


def valid_teacher_json(bundle):
    return {
        "skip": False, "type": bundle["type"], "user_task": bundle["user_task"],
        "question_user": "Which center maintains the related instrument record?",
        "question_source": "What does the record say about the related instrument?",
        "expected_behaviour": "answer", "reference_answer": "Center Beta maintains the related instrument record.",
        "keypoints": [
            {"text": "The record gives the measurement value", "decisive": True, "source": "E1", "quote": "A concrete measurement value is 12 units."},
            {"text": "Center Beta maintains the related instrument record", "decisive": True, "source": "E2", "quote": "The center maintains the related instrument record."},
        ], "pages_used": ["E1", "E2"], "construction_note": "It links a record to its managing center."
    }


class S2WorkflowTests(unittest.TestCase):
    def test_pilot_selection_has_two_bundles_of_each_type(self):
        kinds = ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable")
        bundles = [{"type": kind, "bundle_id": kind} for kind in kinds for _ in range(2)]
        selected = select_pilot(bundles, seed=42)
        self.assertEqual(len(selected), 10)
        self.assertEqual({row["type"] for row in selected}, set(kinds))
        self.assertTrue(all(sum(row["type"] == kind for row in selected) == 2 for kind in kinds))

    def test_generation_skips_try_the_next_seeded_candidate(self):
        bundles = [
            {"type": "multi_hop", "bundle_id": "skip-first", "catalogue_id": "cat-a", "pages": []},
            {"type": "multi_hop", "bundle_id": "usable-second", "catalogue_id": "cat-b", "pages": []},
        ]
        generated = [
            {"id": "skip-first", "type": "multi_hop", "status": "skip", "skip_reason": "Unrelated facts."},
            {"id": "usable-second", "type": "multi_hop", "status": "generated", "question_user": "A supported question?"},
        ]
        writes = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pilot.jsonl"
            with patch("scripts.generate_questions_s2.read_jsonl", return_value=bundles), \
                 patch("scripts.generate_questions_s2.write_jsonl", side_effect=lambda path, rows: writes.update({str(path): rows})), \
                 patch("scripts.generate_questions_s2.generate_one", side_effect=generated):
                rows = generate_pilot(output_path=output, selected_types=["multi_hop"], count_per_type=1)
        self.assertEqual([row["id"] for row in rows], ["usable-second"])
        self.assertEqual(writes[str(output.with_name("generation_skips_s2.jsonl"))][0]["skip_reason"], "Unrelated facts.")

    def test_comparison_excludes_value_named_in_any_selected_title(self):
        self.assertTrue(_comparison_value_in_any_title(["2019", "2021", "2023"], ["Station A (2019)", "Station B", "Station C"]))
        self.assertFalse(_comparison_value_in_any_title(["1.2 m", "2.4 m"], ["Station A", "Station B"]))

    def test_comparison_requires_parseable_comparable_values_and_allowed_relation(self):
        self.assertIsNotNone(_parse_comparison_value("Start time", "1981-01-01 to 1983-12-31"))
        self.assertIsNotNone(_parse_comparison_value("Depth", "12 m"))
        self.assertIsNone(_parse_comparison_value("Country", "France"))
        pages = [
            {"fields": {"Country": "France"}},
            {"fields": {"Country": "France"}},
            {"fields": {"Country": "France"}},
        ]
        self.assertIsNone(_shared_comparison_relation(pages))

    def test_unanswerable_catalogue_note_uses_field_equals_and_sanitizes_markup(self):
        cat = {"name": "Example", "members": [{"url": "u1", "title": "Record A"}, {"url": "u2", "title": "Record B"}],
               "_second_field": "Station", "_fields": {
                   "u1": {"Country": "France ![map](https://example.test/map)"},
                   "u2": {"Station": "North Site"},
               }}
        note = _note("unanswerable", cat, [], "Country", values=("France", "North Site"))
        self.assertIn("with Country =", note)
        self.assertNotIn("![", note)
        self.assertNotIn("http", note)

    def test_derived_keypoint_must_not_have_quote_and_absence_quote_is_ignored(self):
        bundle = sample_bundle("ambiguity")
        result = {
            "skip": False, "type": "ambiguity", "user_task": bundle["user_task"],
            "question_user": "Which record is this?", "question_source": "Which record is this?",
            "expected_behaviour": "disambiguate", "reference_answer": "Two records.",
            "pages_used": ["E1", "E2"], "construction_note": "The name occurs twice.",
            "keypoints": [{"text": "Two records share a name", "decisive": True, "derived": True,
                           "source": "E1", "quote": "A concrete measurement value is 12 units."}],
        }
        self.assertTrue(any("derived and must not include quotes" in issue for issue in response_issues(result, bundle)))
        absence_bundle = sample_bundle("unanswerable")
        absence_result = {**result, "type": "unanswerable", "user_task": absence_bundle["user_task"],
            "expected_behaviour": "state_absence", "keypoints": [{"text": "No such record exists", "decisive": True,
                "source": "E1", "quote": "not an exact quote"}]}
        self.assertFalse(any("quote" in issue for issue in response_issues(absence_result, absence_bundle)))

    def test_incomplete_teacher_json_retries_once_then_validates(self):
        bundle = sample_bundle()
        responses = [{"json": {"skip": False, "question_user": "incomplete"}, "text": "{}"},
                     {"json": valid_teacher_json(bundle), "text": json.dumps(valid_teacher_json(bundle)), "model": "mock-teacher", "cache_hit": False}]
        calls = []
        def fake_call(*args, **kwargs):
            calls.append(kwargs)
            return responses[len(calls) - 1]
        output = generate_one(bundle, "teacher system prompt", llm_call=fake_call, mock=False)
        self.assertEqual(output["status"], "generated")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(output["teacher_prompt_sha256"]), 64)

    def test_invalid_json_schema_is_not_repaired_or_accepted(self):
        bundle = sample_bundle()
        self.assertTrue(response_issues({"skip": False, "question": "alternate"}, bundle))

    def test_quote_verification_maps_each_decisive_quote_to_its_chunk(self):
        question = {"type": "multi_hop", "keypoints": [
            {"text": "fact one", "decisive": True, "source": "E1", "quote": "A concrete measurement value is 12 units."},
            {"text": "fact two", "decisive": True, "source": "E2", "quote": "The center maintains the related instrument record."},
        ], "pages": sample_bundle()["pages"]}
        result = quote_verbatim_check(question, {"c1": question["pages"][0]["text"], "c2": question["pages"][1]["text"]})
        self.assertTrue(result["passed"])
        self.assertEqual(question["keypoints"][0]["gold_chunk_ids"], ["c1"])
        self.assertTrue(multi_page_check(question)["passed"])

    def test_series_completeness_accepts_member_name_aliases(self):
        question = {
            "type": "set_all", "target_field": "station", "target_value": "station value",
            "catalogue_name": "Test catalogue", "field_match_urls": ["https://example.org/a", "https://example.org/b"],
            "expected_items": ["ETC L2 ARCHIVE from Monte Bondone", "ETC L2 ARCHIVE from Renon"],
            "reference_answer": "The records are from Monte Bondone and Renon.", "keypoints": [],
        }
        chunks = [{"chunk_id": "c1", "url": "https://example.org/a", "text": "station value"}]
        catalogue_fields = [
            {"url": "https://example.org/a", "record_name": "ETC L2 ARCHIVE from Monte Bondone", "fields": {"station": "station value"}},
            {"url": "https://example.org/b", "record_name": "ETC L2 ARCHIVE from Renon", "fields": {"station": "station value"}},
        ]
        with patch("scripts.check_questions_s2.read_jsonl", return_value=catalogue_fields):
            result = set_completeness_check(question, chunks, _build_bm25_index(chunks))
        self.assertTrue(result["passed"])

    def test_abstention_request_years_are_not_answer_leakage_or_template_values(self):
        quote = "Stop time (UTC)\n2025-01-01 01:00:00"
        question = {
            "expected_behaviour": "state_absence", "question_user": "Is there a 2025–2026 record?",
            "question_source": "Does a record cover 2025–2026?", "reference_answer": "No record covers 2025–2026.",
            "pages": [{"url": "https://example.org/gold", "title": "Record Alpha"}],
            "keypoints": [{"text": "The requested record is absent", "quote": quote}],
        }
        checks = _rules(question, {
            "https://example.org/gold": quote,
            "https://example.org/other-1": quote,
            "https://example.org/other-2": quote,
            "https://example.org/other-3": quote,
        })
        self.assertTrue(checks["question_does_not_include_answer"]["passed"])
        self.assertTrue(checks["not_template_text"]["passed"])

    def test_absence_keypoint_without_quote_is_valid_teacher_output(self):
        bundle = sample_bundle("unanswerable")
        candidate = valid_teacher_json(bundle)
        candidate["type"] = "unanswerable"
        candidate["expected_behaviour"] = "state_absence"
        candidate["keypoints"] = [{"text": "No record with both requested properties is available", "decisive": True, "sources": [], "quotes": []}]
        self.assertEqual(response_issues(candidate, bundle), [])

    def test_quote_support_checks_multi_page_sources_and_only_rejects_decisive_failure(self):
        question = {"expected_behaviour": "answer", "pages": sample_bundle()["pages"], "keypoints": [
            {"text": "The answer combines both facts", "decisive": True, "sources": ["E1", "E2"],
             "quotes": ["A concrete measurement value is 12 units.", "The center maintains the related instrument record."]},
            {"text": "An unsupported context detail", "decisive": False, "source": "E1", "quote": "not present"},
        ]}
        result = _quote_support_check(question, Path("/private/tmp"), mock=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["non_decisive_failures"], [2])

    def test_absence_claim_is_exempt_from_quote_support(self):
        question = {"expected_behaviour": "state_absence", "pages": sample_bundle()["pages"], "keypoints": [
            {"text": "No such record is available", "decisive": True, "sources": [], "quotes": []}]}
        result = _quote_support_check(question, Path("/private/tmp"), mock=True)
        self.assertTrue(result["passed"])
        self.assertFalse(result["keypoints"][0]["applicable"])


if __name__ == "__main__":
    unittest.main()
