"""Mock tests for reviewed-question application and scoring outputs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.check_questions import _safe_review_export_path, _write_xlsx_minimal
from scripts.prepare_reviewed_questions import prepare_reviewed_questions
from scripts.score_answers import keypoint_metrics, needs_manual_check, score_answer_record


class ScoringWorkflowTests(unittest.TestCase):
    def test_blank_review_decisions_stop_and_list_question_ids(self):
        with self.assertRaisesRegex(ValueError, "W013, W021"):
            prepare_reviewed_questions([{"id": "W013"}, {"id": "W021"}], {})

    def test_review_export_preserves_a_workbook_with_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "review.xlsx"
            _write_xlsx_minimal(pd.DataFrame([
                {"id": "W013", "decision (accept/edit/reject)": "accept"},
            ]), target)
            alternate = _safe_review_export_path(target)
            self.assertNotEqual(target, alternate)
            self.assertTrue(target.exists())
            self.assertFalse(alternate.exists())

    def test_keypoint_metrics_use_decisive_correctness_and_all_keypoints_completeness(self):
        points = [
            {"text": "decisive", "decisive": True},
            {"text": "secondary", "decisive": False},
        ]
        result = keypoint_metrics(points, ["Entailment", "Neutral"])
        self.assertTrue(result["correctness"])
        self.assertEqual(result["completeness"], 0.5)
        self.assertFalse(keypoint_metrics(points, ["Neutral", "Entailment"])["correctness"])
        self.assertFalse(keypoint_metrics(points, ["Entailment", "Contradiction"])["correctness"])

    def test_manual_check_flag_uses_the_two_disagreement_thresholds(self):
        self.assertTrue(needs_manual_check(True, 3))
        self.assertFalse(needs_manual_check(True, 4))
        self.assertTrue(needs_manual_check(False, 4))
        self.assertFalse(needs_manual_check(False, 3))

    def test_scoring_record_calls_only_requested_metrics_and_marks_refusal_faithfulness_nan(self):
        question = {
            "question": "Question?", "reference_answer": "Reference",
            "keypoints": [{"text": "decisive point", "decisive": True}],
        }
        answer = {
            "id": "W1", "type": "list_all", "question": "Question?", "reference_answer": "Reference",
            "condition": "C1", "gold_position": "first", "answer": "A supported answer.",
            "passage_map": [],
        }
        judge_calls = []
        ragchecker_calls = []
        ragas_calls = []

        def judge(system, user):
            judge_calls.append((system, user))
            if system == "flask":
                return {"text": '{"score":4,"reason":"Accurate."}', "json": {"score": 4, "reason": "Accurate."}}
            return {"text": '{"label":"answers","reason":"It responds."}', "json": {"label": "answers", "reason": "It responds."}}

        def ragchecker(row, q):
            ragchecker_calls.append(row["condition"])
            return ["Entailment"], [{"prompt": "mock prompt", "raw_response": "Entailment"}]

        def ragas(metric, row, q):
            ragas_calls.append(metric)
            return {"value": 0.75, "status": "scored"}, {"raw_calls": ["mock raw"]}

        scored = score_answer_record(
            answer, question, {}, flask_system="flask", abstention_system="abstention",
            judge_json=judge, ragchecker_fn=ragchecker, ragas_fn=ragas,
        )
        self.assertTrue(scored["ragchecker"]["correctness"])
        self.assertEqual(scored["flask"]["score"], 4)
        self.assertEqual(scored["abstention"]["label"], "answers")
        self.assertEqual(scored["faithfulness"]["value"], 0.75)
        self.assertEqual(ragchecker_calls, ["C1"])
        self.assertEqual(ragas_calls, ["faithfulness"])

        answer["condition"] = "C3"
        answer["answer"] = "The knowledge base does not provide this information."
        def abstain_judge(system, user):
            return {"text": '{"label":"abstains"}', "json": {"label": "abstains"}}
        scored = score_answer_record(
            answer, question, {}, flask_system="flask", abstention_system="abstention",
            judge_json=abstain_judge, ragchecker_fn=ragchecker, ragas_fn=ragas,
        )
        self.assertEqual(scored["faithfulness"]["status"], "NaN")
        self.assertIsNone(scored["ragchecker"])
        self.assertIsNone(scored["flask"])


if __name__ == "__main__":
    unittest.main()
