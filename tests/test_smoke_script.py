"""Mock tests for the live API smoke-test request builder."""

import unittest
from unittest.mock import patch

from scripts.smoke_test import smoke_role


class SmokeRoleTests(unittest.TestCase):
    @patch("scripts.smoke_test.call_llm")
    def test_judge_ping_uses_json_and_disables_qwen_thinking(self, mock_call):
        mock_call.return_value = {
            "json": {"ok": True, "role": "judge"},
            "usage": {"total_tokens": 12},
            "cache_hit": False,
        }
        settings = {
            "model": "qwen3.7-plus",
            "base_url": "https://example.test/v1",
            "api_key_env": "TEST_API_KEY",
        }

        result = smoke_role("judge", settings)

        self.assertEqual(result["json"]["role"], "judge")
        kwargs = mock_call.call_args.kwargs
        self.assertTrue(kwargs["json_mode"])
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})

    @patch("scripts.smoke_test.call_llm")
    def test_teacher_ping_rejects_wrong_json_role(self, mock_call):
        mock_call.return_value = {"json": {"ok": True, "role": "generator"}}
        with self.assertRaises(ValueError):
            smoke_role("teacher", {
                "model": "openai/gpt-6-sol",
                "base_url": "https://example.test/v1",
                "api_key_env": "TEST_API_KEY",
            })


if __name__ == "__main__":
    unittest.main()
