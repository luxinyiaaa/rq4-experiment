from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.metrics_v2 import (
    _call_json,
    _factual_claim_output,
    citation_validity,
    keypoint_summary,
    metric_applicability,
    score_one,
    terminal_pilot_summary,
)


class MetricsV2Tests(unittest.TestCase):
    def test_json_output_parsing_uses_mock_llm(self):
        def mock_llm(role, messages, **kwargs):
            return {"json": {"score": 4, "reason": "It answers the request."}, "text": "{}", "usage": {"total_tokens": 9}}

        result = _call_json("judge_primary", "response_relevance", {"question": "q", "answer": "a"}, mock_llm=mock_llm)
        self.assertEqual(result["json"]["score"], 4)
        self.assertTrue(result["json"]["reason"])

    def test_metric_applicability_exclusions(self):
        self.assertFalse(metric_applicability("answer_correctness", "C3", "answers")[0])
        self.assertFalse(metric_applicability("evidence_faithfulness", "C1", "abstains")[0])
        self.assertTrue(metric_applicability("retrieved_evidence_coverage", "C4", "answers")[0])
        self.assertFalse(metric_applicability("retrieved_evidence_coverage", "C2", "answers")[0])

    def test_invalid_citation_ids_fail_program_check(self):
        result = citation_validity("Claim [P1]. Another claim [P4].", {"P1", "P2"})
        self.assertEqual(result, {"valid": False, "found_ids": ["P1", "P4"], "invalid_ids": ["P4"]})

    def test_keypoint_outputs_are_parsed_and_aggregated(self):
        points = [{"text": "x", "decisive": True}, {"text": "y", "decisive": False}]
        self.assertEqual(keypoint_summary(points, ["Entailment", "Neutral"]), {
            "correct": True, "completeness": 0.5, "labels": ["Entailment", "Neutral"]
        })
        with self.assertRaises(ValueError):
            keypoint_summary(points, ["Entailment"])

    def test_call_llm_disk_cache_is_used_with_mock_client(self):
        from rq4 import llm
        with tempfile.TemporaryDirectory() as temp:
            root = __import__("pathlib").Path(temp)
            config = root / "models.yaml"
            config.write_text(
                "judge_primary:\n  model: mock-model\n  base_url: https://mock.invalid/v1\n  api_key_env: METRICS_TEST_KEY\n  thinking_mode: off\n",
                encoding="utf-8",
            )
            count = {"calls": 0}

            class FakeCompletions:
                def create(self, **kwargs):
                    count["calls"] += 1
                    message = SimpleNamespace(content='{"ok": true}')
                    choice = SimpleNamespace(message=message)
                    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7, model_dump=lambda: {})
                    return SimpleNamespace(choices=[choice], usage=usage, model="mock-model")

            fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
            with patch.dict(os.environ, {"METRICS_TEST_KEY": "not-a-real-key"}):
                first = llm.call_llm(
                    "judge_primary", [{"role": "user", "content": "test"}], json_mode=True,
                    config_path=config, cache_dir=root / "cache", log_dir=root / "logs", client=fake_client,
                )
                second = llm.call_llm(
                    "judge_primary", [{"role": "user", "content": "test"}], json_mode=True,
                    config_path=config, cache_dir=root / "cache", log_dir=root / "logs",
                )
            self.assertEqual(first["json"], {"ok": True})
            self.assertTrue(second["cache_hit"])
            self.assertEqual(count["calls"], 1)

    def test_terminal_summary_cannot_echo_item_details(self):
        rendered = "\n".join(terminal_pilot_summary(5, 100, 7, 150))
        self.assertNotIn("W001", rendered)
        self.assertNotIn("question text", rendered)
        self.assertNotIn("BLIND_SENTINEL", rendered)

    def test_score_one_pipeline_runs_with_mock_judges(self):
        answer = {
            "id": "TEST01", "condition": "C1", "type": "paraphrased", "answer": "The value is 7 [P1].",
            "passages": [{"passage_id": "P1", "text": "The value is 7."}],
        }
        question = {
            "id": "TEST01", "question": "What is the value?", "reference_answer": "7",
            "keypoints": [{"text": "The value is 7", "decisive": True}], "url": "https://example.test/page",
        }
        def json_mock(role, prompt_name, payload, **kwargs):
            outputs = {
                "response_relevance": {"score": 5, "reason": "It directly answers the question."},
                "appropriate_abstention": {"label": "answers", "reason": "It supplies the requested value."},
                "statement_decomposition": {"statements": [{"statement": "The value is 7 [P1].", "cited_passage_ids": ["P1"], "claims": ["The value is 7."]}], "reason": "One answer sentence contains one claim."},
                "factual_accuracy": {"label": "supported", "reason": "The page states this value."},
                "citation_support": {"entailed": True, "reason": "The cited passage entails the statement."},
                "citation_completeness": {"statements": [{"needs_attribution": True, "reason": "The statement uses a fact from the passage."}]},
            }
            return {"json": outputs[prompt_name], "usage": {"total_tokens": 5}, "api_model": "mock-model"}

        async def ragas_mock(*args, **kwargs):
            return {"value": 1.0, "status": "scored", "reason": "One of one claims is supported.", "calls": []}

        ragchecker = {"values": ["Entailment"], "trace": [{"label": "Entailment", "reason": "Supported."}]}
        with patch("scripts.metrics_v2._call_json", side_effect=json_mock), \
             patch("scripts.metrics_v2._ragas_score", side_effect=ragas_mock), \
             patch("scripts.metrics_v2._ragchecker_run", return_value=ragchecker):
            scored = score_one(
                answer, question,
                settings_by_role={"judge_primary": {}, "judge_secondary": {}},
                pages={"https://example.test/page": "The value is 7."},
            )
        self.assertEqual(scored["metrics"]["answer_correctness"]["primary"], True)
        self.assertEqual(scored["metrics"]["response_relevance"]["secondary"]["score"], 5)
        self.assertEqual(scored["metrics"]["citation_support"]["primary"]["score"], 1.0)

    def test_primary_abstention_controls_applicability_for_both_judges(self):
        answer = {"id": "TEST02", "condition": "C1", "answer": "I cannot determine this.", "passages": []}
        question = {"id": "TEST02", "question": "What is the value?", "reference_answer": "7",
                    "keypoints": [{"text": "The value is 7", "decisive": True}], "url": "https://example.test/page"}
        calls = []
        def json_mock(role, prompt_name, payload, **kwargs):
            calls.append((role, prompt_name))
            if prompt_name == "response_relevance": return {"json": {"score": 2, "reason": "It does not answer."}}
            if prompt_name == "appropriate_abstention":
                label = "abstains" if role == "judge_primary" else "answers"
                return {"json": {"label": label, "reason": "Mock decision."}}
            if prompt_name == "instruction_fulfilment": return {"json": {"met": True, "reason": "It states the evidence limit."}}
            raise AssertionError(prompt_name)
        ragchecker = {"values": ["Neutral"], "trace": []}
        with patch("scripts.metrics_v2._call_json", side_effect=json_mock), \
             patch("scripts.metrics_v2._ragas_score") as ragas_mock, \
             patch("scripts.metrics_v2._ragchecker_run", return_value=ragchecker):
            scored = score_one(answer, question, settings_by_role={"judge_primary": {}, "judge_secondary": {}}, pages={})
        ragas_mock.assert_not_called()
        self.assertEqual(scored["metrics"]["appropriate_abstention"]["primary"]["label"], "abstains")
        self.assertEqual(scored["metrics"]["appropriate_abstention"]["secondary"]["label"], "answers")
        self.assertEqual(scored["metrics"]["evidence_faithfulness"]["primary"]["status"], "excluded")
        self.assertEqual(scored["metrics"]["evidence_faithfulness"]["secondary"]["status"], "excluded")
        self.assertEqual(sum(name == "instruction_fulfilment" for _, name in calls), 2)

    def test_factual_output_accepts_single_result_wrapper(self):
        wrapped = {"supported_share": 1.0, "cannot_be_determined_share": 0.0,
                   "results": [{"label": "supported", "reason": "The page states this claim."}]}
        self.assertEqual(_factual_claim_output(wrapped), wrapped["results"][0])


if __name__ == "__main__":
    unittest.main()
