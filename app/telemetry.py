"""Optional Langfuse spans. Telemetry setup failures do not stop Q&A."""

import logging
from contextlib import ExitStack, contextmanager
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)
_active_client = None


@lru_cache
def get_tracing_client():
    global _active_client
    settings = get_settings()
    if not settings.tracing_enabled or not (
        settings.langfuse_public_key and settings.langfuse_secret_key.get_secret_value()
        and settings.langfuse_base_url
    ):
        return None
    try:
        from langfuse import Langfuse

        _active_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            timeout=5, flush_interval=1,
        )
        return _active_client
    except Exception:
        logger.warning("Tracing could not initialize; continuing without it.")
        return None


@contextmanager
def observation(name: str, *, input: dict, kind: str = "span", **kwargs):
    client = get_tracing_client()
    stack = ExitStack()
    span = None
    if client is not None:
        try:
            span = stack.enter_context(client.start_as_current_observation(
                name=name, as_type=kind, input=input, **kwargs,
            ))
        except Exception:
            logger.warning("Tracing span could not start; continuing without it.")
    # Application exceptions must propagate; do not catch errors around this yield.
    with stack:
        yield span


def update_observation(span, **kwargs):
    if span is not None:
        try:
            span.update(**kwargs)
        except Exception:
            logger.warning("Tracing update failed.")


def current_trace_id() -> str | None:
    client = get_tracing_client()
    try:
        return client.get_current_trace_id() if client else None
    except Exception:
        return None


def shutdown_tracing():
    if _active_client is not None:
        try:
            _active_client.shutdown()
        except Exception:
            logger.warning("Tracing shutdown failed.")
