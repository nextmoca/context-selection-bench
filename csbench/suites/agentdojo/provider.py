"""Gemini via API key for AgentDojo's Google LLM element.

The installed AgentDojo release builds a Vertex AI client and only accepts its
legacy Gemini enum values. This module builds the LLM element the harness
hands to ``PipelineConfig(llm=...)`` instead:

- the API key comes from ``GEMINI_API_KEY`` or ``NM_GEMINI_API_KEY`` and is
  never logged, stored on the element, or written to any artifact;
- the client is an API-key client (not Vertex) with a 60 s per-request timeout
  and a single SDK attempt, so AgentDojo's own three-attempt retry wrapper
  stays in control of per-call retries;
- the legacy identifier AgentDojo accepts is mapped to the model actually
  requested, and both are recorded in the run manifest (``ModelIdentity``).

The element's ``name`` is the legacy identifier, because AgentDojo derives the
attack's model wording and the result paths from the pipeline name; keeping it
identical in both arms keeps the attack text identical in both arms.

Both factories are injectable so tests run without the SDK or the network.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from .config import ModelIdentity

MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
}


class ProviderConfigError(RuntimeError):
    """The provider cannot be configured (missing key, unknown alias)."""


def resolve_provider_model(agentdojo_model: str, provider_model: Optional[str] = None) -> str:
    """The model to request from the provider for an AgentDojo identifier."""
    if provider_model:
        return provider_model
    try:
        return MODEL_ALIASES[agentdojo_model]
    except KeyError:
        raise ProviderConfigError(
            f"no provider model alias for {agentdojo_model!r}; set model.provider_model explicitly"
        ) from None


def read_api_key(env_names: list[str] | tuple[str, ...], environ: Optional[dict[str, str]] = None) -> str:
    """Return the first non-empty key named by ``env_names``. Never log it."""
    env = os.environ if environ is None else environ
    for name in env_names:
        value = env.get(name)
        if value:
            return value
    raise ProviderConfigError(f"no API key found; set one of {list(env_names)}")


def default_client_factory(api_key: str, *, timeout_s: float, attempts: int) -> Any:
    """API-key ``google-genai`` client with a hard per-request timeout and
    SDK retries disabled beyond one attempt."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=max(1, int(attempts))),
        ),
    )


def default_llm_factory(provider_model: str, client: Any, *, temperature: float, max_tokens: int) -> Any:
    from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM

    return GoogleLLM(provider_model, client, temperature=temperature, max_tokens=max_tokens)


def build_llm_element(
    model: ModelIdentity,
    *,
    environ: Optional[dict[str, str]] = None,
    client_factory: Callable[..., Any] = default_client_factory,
    llm_factory: Callable[..., Any] = default_llm_factory,
) -> Any:
    """Build the LLM pipeline element for ``model``.

    Returns an element whose ``name`` is ``model.agentdojo_model`` and whose
    requests go to ``model.provider_model``. The key is passed to the client
    factory only; it is not retained on the returned element.
    """
    if model.provider != "google":
        raise ProviderConfigError(f"this provider module handles 'google' only, got {model.provider!r}")
    provider_model = resolve_provider_model(model.agentdojo_model, model.provider_model)
    api_key = read_api_key(model.api_key_env, environ)
    client = client_factory(api_key, timeout_s=model.request_timeout_s, attempts=model.sdk_attempts)
    del api_key
    llm = llm_factory(provider_model, client, temperature=model.temperature, max_tokens=model.max_output_tokens)
    llm.name = model.agentdojo_model
    return llm


def describe_model(model: ModelIdentity) -> dict[str, Any]:
    """Manifest-safe description: both identifiers, parameters, key *names*."""
    return {
        "provider": model.provider,
        "agentdojo_model": model.agentdojo_model,
        "provider_model": resolve_provider_model(model.agentdojo_model, model.provider_model),
        "client_package": model.client_package,
        "temperature": model.temperature,
        "max_output_tokens": model.max_output_tokens,
        "request_timeout_s": model.request_timeout_s,
        "sdk_attempts": model.sdk_attempts,
        "api_key_env": list(model.api_key_env),
    }
