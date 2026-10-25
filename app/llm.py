"""Small Gemini REST client with validated JSON output and sanitized errors."""

import json
import re
import time
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.errors import ServiceError
from app.telemetry import observation, update_observation
from app.verify_access import request_json


class GeminiClient:
    @staticmethod
    def _request(url, headers, body):
        # One transport retry for temporary provider outages. This is separate
        # from the graph's two query rewrites and never retries quota errors.
        for attempt in range(2):
            try:
                return request_json(url, headers, body)
            except HTTPError as error:
                if error.code not in {500, 502, 503, 504} or attempt == 1:
                    raise
            except (URLError, TimeoutError):
                if attempt == 1:
                    raise
            time.sleep(1)

    def generate(self, instruction: str, data: dict, schema: type[BaseModel], *, task: str):
        settings = get_settings()
        model = settings.gemini_model.removeprefix("models/")
        key = settings.gemini_api_key.get_secret_value()
        if not key or not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise ServiceError("Set a valid GEMINI_API_KEY and GEMINI_MODEL in .env.", 503)
        body = {
            "systemInstruction": {"parts": [{"text": instruction}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(data)}]}],
            "generationConfig": {
                "temperature": 0.1, "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema.model_json_schema(),
            },
        }
        with observation(f"gemini.{task}", input=data, kind="generation", model=model) as span:
            try:
                result = self._request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    {"x-goog-api-key": key}, body,
                )
            except HTTPError as error:
                if error.code == 429:
                    raise ServiceError("Gemini quota or rate limit reached. Check AI Studio usage and try again later.", 429) from None
                if error.code in {400, 401, 403, 404}:
                    raise ServiceError("Gemini rejected the request. Check the configured key, model, and project permissions.", 502) from None
                raise ServiceError("Gemini is temporarily unavailable.", 502) from None
            except (URLError, TimeoutError):
                raise ServiceError("Could not reach Gemini before the request timed out.", 504) from None
            except (ValueError, OSError):
                raise ServiceError("Gemini returned an unreadable response.", 502) from None
            try:
                candidates = result.get("candidates", [])
                if not candidates or candidates[0].get("finishReason") != "STOP":
                    raise ValueError("Incomplete or blocked output")
                text = "".join(
                    part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                    if not part.get("thought")
                )
                parsed = schema.model_validate_json(text)
            except (ValueError, TypeError, KeyError, ValidationError):
                raise ServiceError("Gemini returned incomplete or invalid structured output. Please try again.", 502) from None
            usage = result.get("usageMetadata", {})
            update_observation(span, output=parsed.model_dump(), usage_details={
                "input": usage.get("promptTokenCount", 0),
                "output": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            })
            return parsed
