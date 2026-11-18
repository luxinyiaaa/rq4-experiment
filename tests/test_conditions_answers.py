"""Mock tests for condition construction and answer generation."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.build_conditions import build_question_conditions
from scripts.generate_answers import generate_answers, review_frame


class FakeIndex:
    def __init__(self):
        self.window_records = []
        for index in range(16):
            self.window_records.append({
                "window_id": f"D{index}",
                "url": f"https://data.test/datasets/{index}",
                "domain": "data.test",
                "url_template": "data.test/datasets",
                "page_type": "dataset_record",
                "chunk_ids": [f"c{index}"],
                "text": f"# Candidate {index}\n" + (f"Unique facts for candidate {index}. " * 40),
            })
        self.window_embeddings = np.asarray([[1.0, index / 100] for index in range(16)], dtype=np.float32)
        self.chunk_records = [
            {"chunk_id": "0", "url": "https://data.test/datasets/gold", "text": "# Gold dataset\nOriginal gold facts."},
            *[{
                "chunk_id": str(index + 1),
                "url": row["url"],
                "text": row["text"],
            } for index, row in enumerate(self.window_records)],
        ]

    def _query_embedding(self, text):
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def retrieve(self, query, k=10):
        return [
            {"chunk_id": f"retrieved-{index}", "url": "https://data.test/all", "similarity": 0.9 - index / 100, "text": f"retrieved chunk {index}"}
            for index in range(k)
        ]


def question(qid, kind="list_all"):
    return {
        "id": qid,
        "type": kind,
        "status": "generated",
        "question": f"What facts are listed for {qid}?",
        "reference_answer": "The answer is the gold fact.",
        "keypoints": [{"text": "the gold fact", "decisive": True}],
        "url": "https://data.test/datasets/gold",
        "page_type": "dataset_record",
        "text": "# Gold dataset\n" + ("Original gold facts. " * 40),
        "chunk_ids": ["0"],
        "gold_chunk_ids": ["0"],
        "partner": None,
        "global_conflict": {"conflicts": []},
        "duplicate_title": {"matches": []},
    }


class ConditionAndAnswerTests(unittest.TestCase):
    def test_condition_builder_judges_candidates_and_builds_four_passage_sets(self):
        calls = []

        def fake_judge(role, messages, **kwargs):
            calls.append((role, messages, kwargs))
            content = messages[1]["content"]
            count = content.count("[D")
            return {
                "json": {"labels": [
                    {"passage_id": f"D{position}", "label": "safe", "reason": "No answer to the question."}
                    for position in range(1, 11)
                ]},
                "model": "mock-judge",
                "usage": {"total_tokens": 50},
                "cache_hit": False,
            }

        rows = build_question_conditions([question("Q1")], FakeIndex(), "judge prompt", llm_call=fake_judge)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "judge")
        result = rows[0]
        self.assertEqual(result["gold_position"], "first")
        self.assertEqual(len(result["conditions"]["C1"]["passages"]), 1)
        self.assertEqual(len(result["conditions"]["C2"]["passages"]), 5)
        self.assertEqual(len(result["conditions"]["C3"]["passages"]), 4)
        self.assertEqual(len(result["conditions"]["C4"]["passages"]), 10)
        self.assertEqual(result["conditions"]["C2"]["passages"][0]["role"], "gold")
        self.assertEqual(
            [row["window_id"] for row in result["conditions"]["C2"]["passages"][1:]],
            [row["window_id"] for row in result["conditions"]["C3"]["passages"]],
        )
        self.assertEqual(result["c4_gold_hit_rank"], None)

    def test_gold_position_rotates_in_question_order_within_type(self):
        def fake_judge(role, messages, **kwargs):
            return {"json": {"labels": [
                {"passage_id": f"D{position}", "label": "safe", "reason": "Unrelated."}
                for position in range(1, 11)
            ]}}
        rows = build_question_conditions(
            [question("Q1"), question("Q2"), question("Q3")],
            FakeIndex(), "judge prompt", llm_call=fake_judge,
        )
        self.assertEqual([row["gold_position"] for row in rows], ["first", "middle", "last"])

    def test_answer_generation_uses_one_shared_system_prompt_for_all_conditions(self):
        q = question("Q1")
        q["gold_position"] = "middle"
        q["selected_distractors"] = []
        q["c4_gold_hit_rank"] = 3
        q["distractor_selection"] = {"judge_removed": 0, "prefilter_excluded_count": 0}
        passage = {"passage_id": "P1", "url": "https://data.test/gold", "text": "Gold passage."}
        q["conditions"] = {
            name: {"passages": [passage], "passage_map": [{"passage_id": "P1", "chunk_ids": ["c1"]}]}
            for name in ("C1", "C2", "C3", "C4")
        }
        calls = []

        def fake_generator(role, messages, **kwargs):
            calls.append((role, messages, kwargs))
            return {
                "text": "An unscored answer.", "model": "configured-model", "api_model": "returned-model-id",
                "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                "elapsed_seconds": 0.2, "cache_hit": False,
            }

        outputs = generate_answers([q], "shared system", "prompt-hash", llm_call=fake_generator, progress=False)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(messages[0]["content"] == "shared system" for _, messages, _ in calls))
        self.assertTrue(all(row["model_id"] == "returned-model-id" for row in outputs))
        frame = review_frame([q], outputs)
        self.assertEqual(len(frame), 1)
        self.assertIn("Gold position: middle", frame.iloc[0]["C2_answer (gold position)"])


if __name__ == "__main__":
    unittest.main()
