"""Check configured services without logging credentials or provider error bodies.

Run with: docker compose exec -T api python -m app.verify_access
"""

import base64
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.config import get_settings


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward authentication headers to a redirect destination.
        return None


def request_json(url: str, headers: dict[str, str], body: dict | None = None) -> dict:
    request = Request(
        url,
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with build_opener(NoRedirects()).open(request, timeout=30) as response:
        return json.load(response)


def report_error(service: str, error: Exception) -> None:
    if isinstance(error, HTTPError):
        hints = {
            400: "Check the API key, model name, and request settings.",
            401: "Credentials were rejected; check the key pair and region.",
            403: "Access denied; check key restrictions and project permissions.",
            404: "Check the configured model or regional base URL.",
            429: "Quota or rate limit reached; check your free-tier limits.",
        }
        print(f"FAIL {service}: HTTP {error.code}. "
              f"{hints.get(error.code, 'Service request failed; try again later.')}")
    elif isinstance(error, (URLError, TimeoutError)):
        print(f"FAIL {service}: connection failed or timed out.")
    else:
        print(f"FAIL {service}: unexpected response or configuration error.")


def verify_gemini(settings) -> bool:
    model = settings.gemini_model.removeprefix("models/")
    if not settings.gemini_api_key.get_secret_value() or not model:
        print("FAIL Gemini: set GEMINI_API_KEY and GEMINI_MODEL in .env.")
        return False
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
        print("FAIL Gemini: GEMINI_MODEL must contain a model ID, not a URL.")
        return False
    try:
        result = request_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": settings.gemini_api_key.get_secret_value()},
            {
                "contents": [{"parts": [{"text": "Reply with exactly DOCSPILOT_OK."}]}],
                "generationConfig": {"maxOutputTokens": 256},
            },
        )
        text = "".join(
            part.get("text", "")
            for candidate in result.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        ).strip()
        if not text:
            print("FAIL Gemini: request accepted but no answer text returned.")
            return False
        print(f"PASS Gemini: {model} returned answer text.")
        return True
    except Exception as error:
        report_error("Gemini", error)
        return False


def verify_langfuse(settings) -> bool:
    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key.get_secret_value()
    base_url = settings.langfuse_base_url.rstrip("/")
    if not (public and secret and base_url):
        print("FAIL Langfuse: set its public key, secret key, and base URL in .env.")
        return False
    parsed = urlsplit(base_url)
    allowed_hosts = {
        "cloud.langfuse.com", "us.cloud.langfuse.com",
        "jp.cloud.langfuse.com", "hipaa.cloud.langfuse.com",
    }
    if (parsed.scheme != "https" or parsed.netloc not in allowed_hosts
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        print("FAIL Langfuse: use your project's HTTPS cloud base URL without an API path.")
        return False
    try:
        auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
        result = request_json(
            f"{base_url}/api/public/projects", {"Authorization": f"Basic {auth}"}
        )
        if not result.get("data"):
            print("FAIL Langfuse: no accessible project returned.")
            return False
        print("PASS Langfuse: credentials authenticated and project is accessible.")
        return True
    except Exception as error:
        report_error("Langfuse", error)
        return False


def main() -> int:
    try:
        settings = get_settings()
    except Exception:
        print("FAIL configuration: check .env formatting. Details omitted to protect secrets.")
        return 1
    gemini_ok = verify_gemini(settings)
    langfuse_ok = verify_langfuse(settings)
    return 0 if gemini_ok and langfuse_ok else 1


if __name__ == "__main__":
    sys.exit(main())
