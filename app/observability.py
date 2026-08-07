"""Constructs the Langfuse client at import time so it becomes the process-wide
singleton (`langfuse.get_client()`) before any instrumented OpenAI call can
happen. Import this module first, before anything that touches deepinfra.py —
deepinfra.py itself does this as its very first import for exactly that reason.

If LANGFUSE_PUBLIC_KEY/SECRET_KEY aren't set (or are blank — see tests/conftest.py,
which blanks them so the test suite never sends real traces to a live project),
tracing is explicitly disabled and no network calls are made at all.
"""

from langfuse import Langfuse

from app.config import settings

_has_credentials = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

langfuse_client = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
    tracing_enabled=_has_credentials,
)
