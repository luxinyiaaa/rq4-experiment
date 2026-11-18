"""Unit tests for assignment, quote mapping, and pilot checks."""
import json
import unittest

import numpy as np

from scripts.assign_types import TYPE_COUNTS, assign_records
from scripts.check_questions import (
    check_decisive_labels,
    check_global_conflict,
    find_duplicate_titles,
    mark_near_duplicates,
    page_title_catalog,
    _edited_version,
)
from scripts.generate_questions import (
    PROMPT_PATH, _map_supporting_quotes, generate_one, load_teacher_prompt, select_pilot,
)


class QuestionWorkflowTests(unittest.TestCase):
    def test_teacher_prompt_loads_entire_file_with_all_four_examples(self):
        prompt = load_teacher_prompt(PROMPT_PATH)
        self.assertEqual(prompt, PROMPT_PATH.read_text(encoding="utf-8").strip())
        self.assertIn("## Rules per type", prompt)
        self.assertIn("## Output format", prompt)
        self.assertEqual(sum(f"### Example {i}:" in prompt for i in range(1, 5)), 4)

    def test_assignment_has_requested_counts_and_partners(self):
        windows = []
        for index in range(23):
            windows.append({
                "window_id": f"D{index}", "page_type": "dataset_record",
                "url": f"https://data.example/datasets/{index}", "domain": "data.example",
                "text": f"Dataset facts for item {index}.", "chunk_ids": [str(index)],
            })
        for index in range(27):
            windows.append({
                "window_id": f"A{index}", "page_type": "about_science", 
                "url": f"https://science.example/about/{index}", "domain": "science.example",
                "text": f"Science fact {index}.", "chunk_ids": [str(index)],
            })

        def candidates_for(window_id, n):
            source = next(item for item in windows if item["window_id"] == window_id)
            if source["page_type"] == "dataset_record":
                partner_id = "D1" if window_id != "D1" else "D0"
                partner = next(item for item in windows if item["window_id"] == partner_id)
                return [{**partner, "url_template": "data.example/datasets"}]
            return [{
                "window_id": "candidate", "url": "https://data.example/datasets/candidate",
                "domain": "data.example", "page_type": "dataset_record",
                "url_template": "data.example/datasets", "text": "Related passage", "chunk_ids": ["x"],
            }]

        plan = assign_records(windows, candidates_for, candidate_count=3, seed=42)

        self.assertEqual({kind: sum(row["type"] == kind for row in plan) for kind in TYPE_COUNTS}, TYPE_COUNTS)
        self.assertTrue(all("partner" in row for row in plan if row["type"] == "comparison"))
        self.assertTrue(all("similar" in row for row in plan if row["type"] == "false_premise"))
        self.assertEqual([row["id"] for row in plan], [row["window_id"] for row in windows])

    def test_pilot_sampling_has_3_2_3_2_quota(self):
        plan = []
        for kind, count in TYPE_COUNTS.items():
            plan.extend({"id": f"{kind}-{index}", "type": kind} for index in range(count))
        pilot = select_pilot(plan, seed=42)
        self.assertEqual(
            {kind: sum(row["type"] == kind for row in pilot) for kind in ("list_all", "comparison", "paraphrased", "false_premise")},
            {"list_all": 3, "comparison": 2, "paraphrased": 3, "false_premise": 2},
        )

    def test_quote_exact_match_maps_character_span_to_chunk_id(self):
        item = {"text": "Alpha value", "chunk_ids": ["a", "b"]}
        result, ids, failures = _map_supporting_quotes(
            {"supporting_quotes": [{"source": "passage", "text": "value"}]},
            item,
            {"a": "Alpha ", "b": "value"},
        )
        self.assertEqual(failures, [])
        self.assertEqual(ids["passage"], ["b"])
        self.assertEqual((result[0]["char_start"], result[0]["char_end"]), (6, 11))

    def test_invalid_teacher_schema_retries_then_keeps_raw_valid_json(self):
        item = {
            "id": "W1", "window_id": "W1", "type": "paraphrased", "url": "https://x.test/",
            "page_type": "about_science", "text": "A fact.", "chunk_ids": ["c1"],
        }
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raw = {"question": "What fact?", "answer": "A fact."}
            else:
                raw = {
                    "skip": False, "type": "paraphrased", "question": "What fact?",
                    "reference_answer": "A fact.",
                    "keypoints": [{"text": "A fact", "decisive": True}],
                    "supporting_quotes": [{"source": "passage", "text": "A fact."}],
                }
            return {"json": raw, "text": json.dumps(raw), "model": "mock", "cache_hit": False}

        result = generate_one(item, {"c1": "A fact."}, system_prompt="system", llm_call=fake_call)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "generated")
        self.assertFalse(result["quote_failed"])
        self.assertEqual(result["gold_chunk_ids"], ["c1"])
        self.assertEqual(result["teacher_raw_json"]["reference_answer"], "A fact.")
        self.assertTrue(calls[0]["log_messages"])
        self.assertEqual(len(calls[0]["prompt_sha256"]), 64)

    def test_teacher_invalid_schema_twice_becomes_generation_failed(self):
        item = {
            "id": "W1", "window_id": "W1", "type": "paraphrased", "url": "https://x.test/",
            "page_type": "about_science", "text": "A fact.", "chunk_ids": ["c1"],
        }
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs)
            raw = {"question": "What fact?", "answer": "A fact."}
            return {"json": raw, "text": json.dumps(raw), "model": "mock"}

        result = generate_one(item, {"c1": "A fact."}, system_prompt="system", llm_call=fake_call)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "generation_failed")
        self.assertIn("reference_answer", result["generation_error"])
        self.assertNotIn("reference_answer", result["teacher_raw_json"])

    def test_round_trip_requires_all_decisive_entailment_and_no_contradiction(self):
        keypoints = [{"text": "a", "decisive": True}, {"text": "b", "decisive": False}]
        self.assertTrue(check_decisive_labels(keypoints, ["Entailment", "Neutral"])["pass"])
        self.assertFalse(check_decisive_labels(keypoints, ["Neutral", "Entailment"])["pass"])
        self.assertFalse(check_decisive_labels(keypoints, ["Entailment", "Contradiction"])["pass"])

    def test_similarity_marks_both_questions_when_cosine_exceeds_threshold(self):
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0], [0.98, 0.2]], dtype=np.float32)

        questions = [
            {"id": "Q1", "status": "generated", "question": "first"},
            {"id": "Q2", "status": "generated", "question": "near duplicate"},
        ]
        marked = mark_near_duplicates(questions, Encoder(), "prefix: ")
        self.assertTrue(all(row["near_duplicate"] for row in marked))
        self.assertEqual(marked[0]["near_duplicate_matches"][0]["id"], "Q2")

    def test_title_catalog_skips_generic_heading_and_finds_exact_duplicates(self):
        chunks = [
            {"url": "https://x.test/a", "chunk_id": "1", "text": "# EDMED\n# Shared title"},
            {"url": "https://x.test/b", "chunk_id": "2", "text": "# Shared title"},
            {"url": "https://x.test/c", "chunk_id": "3", "text": "# Home\n# Unique"},
        ]
        titles, by_title = page_title_catalog(chunks)
        self.assertEqual(titles["https://x.test/a"], "Shared title")
        self.assertEqual(titles["https://x.test/c"], "Unique")
        result = find_duplicate_titles(
            {"url": "https://x.test/a"}, titles, by_title
        )
        self.assertTrue(result["duplicate_title"])
        self.assertEqual(result["matches"][0]["url"], "https://x.test/b")

    def test_global_conflict_uses_top_ten_then_removes_gold_and_partner_pages(self):
        class Index:
            def retrieve(self, query, k):
                self.query, self.k = query, k
                return [
                    {"chunk_id": str(i), "url": f"https://x.test/{i}", "text": f"passage {i}"}
                    for i in range(10)
                ]

        index = Index()
        calls = []

        def fake_llm(role, messages, **kwargs):
            calls.append((role, messages, kwargs))
            return {"json": {"conflict": True, "conflicts": [{"passage_id": "P1", "what_differs": "a value differs"}]}}

        question = {
            "question": "Question?", "reference_answer": "Answer", "url": "https://x.test/0",
            "partner": {"url": "https://x.test/1"},
            "keypoints": [{"text": "fact", "decisive": True}],
        }
        result = check_global_conflict(question, index, "judge prompt", llm_call=fake_llm)
        self.assertEqual(index.k, 10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["checked_passage_count"], 8)
        self.assertEqual(result["conflicts"][0]["chunk_id"], "2")
        self.assertTrue(calls[0][2]["log_messages"])

    def test_edit_uses_only_three_edit_columns_not_notes(self):
        from scripts.check_questions import (
            MODIFIED_ANSWER, MODIFIED_KEYPOINTS, MODIFIED_QUESTION, REVIEW_DECISION,
        )
        question = {"question": "old", "reference_answer": "old answer", "keypoints": []}
        question["human_review"] = {
            REVIEW_DECISION: "edit", MODIFIED_QUESTION: "new question",
            MODIFIED_ANSWER: "new answer", MODIFIED_KEYPOINTS: "[D] required point",
            "review_note": "ignore this fallback text",
        }
        edited = _edited_version(question)
        self.assertEqual(edited["question"], "new question")
        self.assertEqual(edited["reference_answer"], "new answer")
        self.assertEqual(edited["keypoints"], [{"text": "required point", "decisive": True}])


if __name__ == "__main__":
    unittest.main()
