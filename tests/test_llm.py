"""Mock-based tests for LLM disk caching and JSONL logging."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rq4.llm import call_llm


class LLMCacheAndLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "models.yaml"
        self.config.write_text(
            "teacher:\n  model: test-model\n  base_url: https://example.test/v1\n  api_key_env: TEST_API_KEY\n",
            encoding="utf-8",
        )
        self.old_key = os.environ.get("TEST_API_KEY")
        os.environ["TEST_API_KEY"] = "test-key"
        self.cache = self.root / "cache"
        self.logs = self.root / "logs"

    def tearDown(self):
        if self.old_key is None:
            os.environ.pop("TEST_API_KEY", None)
        else:
            os.environ["TEST_API_KEY"] = self.old_key
        self.temp.cleanup()

    @staticmethod
    def response(content, prompt_tokens=7, completion_tokens=3):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model="test-model-returned-id",
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def test_success_is_cached_and_each_call_is_logged(self):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.response('{"answer": "ok"}')
        args = dict(
            role="teacher",
            messages=[{"role": "user", "content": "Return JSON."}],
            json_mode=True,
            config_path=self.config,
            cache_dir=self.cache,
            log_dir=self.logs,
            client=mock_client,
            log_messages=True,
        )

        first = call_llm(**args)
        second = call_llm(**args)

        self.assertEqual(first["json"], {"answer": "ok"})
        self.assertEqual(first["api_model"], "test-model-returned-id")
        self.assertEqual(second["api_model"], "test-model-returned-id")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

        log_lines = (self.logs / "teacher.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 2)
        first_log, second_log = map(json.loads, log_lines)
        self.assertEqual(first_log["token_usage"]["total_tokens"], 10)
        self.assertEqual(first_log["messages"], args["messages"])
        self.assertIn("elapsed_seconds", first_log)
        self.assertTrue(second_log["cache_hit"])
        self.assertEqual(second_log["token_usage"]["total_tokens"], 10)

    def test_prompt_hash_changes_cache_key(self):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.response('{"answer": "ok"}')
        args = dict(
            role="teacher",
            messages=[{"role": "user", "content": "Return JSON."}],
            json_mode=True,
            config_path=self.config,
            cache_dir=self.cache,
            log_dir=self.logs,
            client=mock_client,
        )
        call_llm(**args, prompt_sha256="old-prompt")
        call_llm(**args, prompt_sha256="new-prompt")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_empty_reasoning_token_count_does_not_fail_successful_response(self):
        mock_client = Mock()
        response = self.response('{"answer": "ok"}')
        response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens="")
        mock_client.chat.completions.create.return_value = response
        result = call_llm(
            role="teacher",
            messages=[{"role": "user", "content": "Return JSON."}],
            json_mode=True,
            config_path=self.config,
            cache_dir=self.cache,
            log_dir=self.logs,
            client=mock_client,
        )
        self.assertEqual(result["json"], {"answer": "ok"})
        self.assertNotIn("reasoning_tokens", result["usage"])

    def test_invalid_json_is_retried_once_then_success_is_cached(self):
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            self.response("not json"),
            self.response('{"answer": "fixed"}'),
        ]
        result = call_llm(
            role="teacher",
            messages=[{"role": "user", "content": "Return JSON."}],
            json_mode=True,
            config_path=self.config,
            cache_dir=self.cache,
            log_dir=self.logs,
            client=mock_client,
        )
        self.assertEqual(result["json"], {"answer": "fixed"})
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        log = json.loads((self.logs / "teacher.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(log["attempts"], 2)
        self.assertEqual(log["status"], "success")

    def test_invalid_json_after_retry_is_not_cached(self):
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            self.response("not json"),
            self.response("still not json"),
        ]
        with self.assertRaises(json.JSONDecodeError):
            call_llm(
                role="teacher",
                messages=[{"role": "user", "content": "Return JSON."}],
                json_mode=True,
                config_path=self.config,
                cache_dir=self.cache,
                log_dir=self.logs,
                client=mock_client,
            )
        self.assertEqual(list(self.cache.glob("*.json")), [])
        log = json.loads((self.logs / "teacher.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(log["status"], "error")

    def test_failed_call_is_not_cached(self):
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RuntimeError("request failed")
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            call_llm(
                role="teacher",
                messages=[{"role": "user", "content": "Hello."}],
                config_path=self.config,
                cache_dir=self.cache,
                log_dir=self.logs,
                client=mock_client,
            )
        self.assertEqual(list(self.cache.glob("*.json")), [])

    def test_concurrent_writes_to_same_cache_key_do_not_collide(self):
        from rq4.llm import _write_cache
        path = self.cache / "shared.json"
        results = [{"text": '{"ok": true}', "usage": {"total_tokens": i}, "model": "mock", "api_model": "mock"}
                   for i in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda result: _write_cache(path, result), results))
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["text"], '{"ok": true}')
        self.assertEqual(len(list(self.cache.glob("*.tmp"))), 0)


if __name__ == "__main__":
    unittest.main()
