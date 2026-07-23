"""Provider-clean Gemini model client for benchmark runs.

Reads credentials and model selection from the environment only. The
``google-genai`` package is an OPTIONAL dependency: it is imported lazily inside
``build_gemini_client`` so this module imports cleanly even when the package is
not installed.

Environment variables:
    GEMINI_API_KEY  -- API key (checked first)
    GOOGLE_API_KEY  -- API key (fallback)
    GEMINI_MODEL    -- model-name override (else DEFAULT_MODEL)
"""

from __future__ import annotations

import os
import time
from typing import Any

from csbench.tokenizing import estimate_tokens

DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _api_key_from_env() -> str:
    """Return the Gemini API key from the environment.

    Checks ``GEMINI_API_KEY`` then ``GOOGLE_API_KEY``. Never accepts a key from
    any other source.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required.")
    return api_key


def resolve_model(model: str | None = None) -> str:
    """Resolve the model name.

    Precedence: explicit ``model`` argument, then ``GEMINI_MODEL`` env var, then
    ``DEFAULT_MODEL``. A leading ``google:`` provider prefix is stripped.
    """
    value = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    return value.split(":", 1)[1] if value.startswith("google:") else value


def build_gemini_client(timeout_ms: int = 60000) -> Any:
    """Build a Gemini client, importing ``google-genai`` lazily.

    The import happens here (not at module load) so that importing this module
    does not require the optional ``google-genai`` dependency.
    """
    from google import genai
    from google.genai import types

    api_key = _api_key_from_env()
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def call_gemini(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    """Call the model with a single prompt and return a normalized result dict.

    Returns a dict with at least ``content``, ``input_tokens``, and
    ``output_tokens``. Token counts come from the response usage metadata when
    present, falling back to an estimate. Temperature is pinned to 0 for
    reproducibility.
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": max_output_tokens},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            text = getattr(response, "text", None) or ""
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            return {
                "content": text,
                "input_tokens": input_tokens or estimate_tokens(prompt),
                "output_tokens": output_tokens or estimate_tokens(text),
                "latency_ms": latency_ms,
            }
        except Exception as exc:  # exercised only by live API transients
            last_exc = exc
            if attempt + 1 >= retries:
                break
            time.sleep(retry_sleep_seconds * (attempt + 1))
    assert last_exc is not None
    raise last_exc
