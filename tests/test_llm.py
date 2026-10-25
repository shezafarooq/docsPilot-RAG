import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from pydantic import SecretStr

from app.errors import ServiceError
from app.llm import GeminiClient
from app.schemas import RewriteDecision


class TestGeminiClient(unittest.TestCase):
    def setUp(self):
        self.settings_patch = patch("app.llm.get_settings", return_value=SimpleNamespace(
            gemini_api_key=SecretStr("private-test-key"), gemini_model="gemini-test"))
        self.trace_patch = patch("app.llm.observation", side_effect=lambda *args, **kwargs: nullcontext(None))
        self.settings_patch.start()
        self.trace_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.addCleanup(self.trace_patch.stop)

    def response(self, text, finish="STOP"):
        return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}

    def generate(self):
        return GeminiClient().generate("Instruction", {"question": "question"}, RewriteDecision, task="rewrite")

    def test_json_is_validated_and_key_goes_in_header_not_url(self):
        with patch("app.llm.request_json", return_value=self.response(json.dumps({"query": "volume mount"}))) as request:
            result = self.generate()
        self.assertEqual(result.query, "volume mount")
        self.assertNotIn("private-test-key", request.call_args.args[0])
        self.assertEqual(request.call_args.args[1]["x-goog-api-key"], "private-test-key")
        self.assertEqual(request.call_args.args[2]["generationConfig"]["responseMimeType"], "application/json")

    def test_malformed_schema_and_truncated_output_fail(self):
        for response in (self.response("not JSON"), self.response('{"query":42}'),
                         self.response('{"query":"partial"}', "MAX_TOKENS"), {"candidates": []}):
            with self.subTest(response=response), patch("app.llm.request_json", return_value=response):
                with self.assertRaises(ServiceError):
                    self.generate()

    def test_quota_error_is_sanitized(self):
        error = HTTPError("https://provider.example", 429, "private-test-key", {}, None)
        with patch("app.llm.request_json", side_effect=error):
            with self.assertRaises(ServiceError) as caught:
                self.generate()
        self.assertEqual(caught.exception.status_code, 429)
        self.assertNotIn("private-test-key", str(caught.exception))

    def test_temporary_outage_retries_once_then_can_succeed(self):
        error = HTTPError("https://provider.example", 503, "Overloaded", {}, None)
        response = self.response('{"query":"volume mount"}')
        with patch("app.llm.request_json", side_effect=[error, response]) as request, patch("app.llm.time.sleep"):
            self.assertEqual(self.generate().query, "volume mount")
        self.assertEqual(request.call_count, 2)

    def test_persistent_outage_stops_after_one_transport_retry(self):
        error = HTTPError("https://provider.example", 503, "Overloaded", {}, None)
        with patch("app.llm.request_json", side_effect=error) as request, patch("app.llm.time.sleep"):
            with self.assertRaises(ServiceError):
                self.generate()
        self.assertEqual(request.call_count, 2)
