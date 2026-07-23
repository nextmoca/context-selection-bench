"""LLMLingua-2 local compression arm.

Wraps the real ``llmlingua.PromptCompressor`` (LLMLingua-2, a small local
xlm-roberta token-classifier) as a context-reduction arm. It concatenates the
request's records into a single context string and compresses that context
down toward the request's token budget, then returns the compressed text as the
rendered context.

``llmlingua`` is an OPTIONAL dependency: it is imported lazily inside the
constructor so that merely importing this module never pulls in the heavy model
stack. Install ``llmlingua`` (and its torch/transformers deps) to use this arm.

The model checkpoint and construction parameters are pinned to the same values
used by the reference benchmark so that published numbers reproduce.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from ..contracts import ContextRequest, ContextResponse
from ..tokenizing import estimate_tokens
from .base import ContextArm

# Real LLMLingua-2 token-classifier checkpoint. ``PromptCompressor``'s own
# default ``model_name`` is a 7B causal LM that is NOT a valid LLMLingua-2
# checkpoint and crashes every call when ``use_llmlingua2=True``, so the model
# name must always be passed explicitly.
DEFAULT_LLMLINGUA2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class Llmlingua2Compressor:
    """Thread-safe wrapper around a single real ``llmlingua.PromptCompressor``.

    The underlying model is a real local model, expensive to load, so exactly
    one instance is typically constructed per process and shared. Compression
    calls are serialized behind a lock because concurrent
    ``compress_prompt`` invocation is not verified thread-safe: correctness is
    guaranteed at the cost of not parallelizing the compression step itself.
    """

    def __init__(self, compressor: Any):
        self._compressor = compressor
        self._lock = threading.Lock()

    def compress(self, context: str, *, rate: float) -> dict[str, Any]:
        with self._lock:
            return self._compressor.compress_prompt(context=[context], rate=rate)


def build_llmlingua2_compressor(
    *,
    model_name: str = DEFAULT_LLMLINGUA2_MODEL,
    device_map: str | None = None,
) -> Llmlingua2Compressor:
    """Construct the real ``llmlingua.PromptCompressor``-backed compressor.

    ``PromptCompressor``'s own default ``device_map`` is ``"cuda"`` (it assumes
    a GPU box); on a CPU-only machine that raises at construction time. Default
    to ``"cpu"`` here, overridable via ``LLMLINGUA2_DEVICE_MAP`` for
    GPU-equipped hosts.

    ``model_name`` is passed explicitly and defaults to a real LLMLingua-2
    checkpoint (see ``DEFAULT_LLMLINGUA2_MODEL``).
    """
    from llmlingua import PromptCompressor  # optional heavy dependency

    resolved_device_map = device_map or os.environ.get("LLMLINGUA2_DEVICE_MAP", "cpu")
    return Llmlingua2Compressor(
        PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=resolved_device_map,
        )
    )


class Llmlingua2Arm(ContextArm):
    """Local LLMLingua-2 compression arm.

    ``select`` concatenates the record texts into a single context, compresses
    it toward ``request.budget.max_context_tokens``, and returns the compressed
    text as the rendered context. The task/question
    (``request.task.prompt``) is NOT part of the records and is not compressed.
    """

    name = "llmlingua2"

    def __init__(
        self,
        compressor: Llmlingua2Compressor | None = None,
        *,
        model_name: str = DEFAULT_LLMLINGUA2_MODEL,
        device_map: str | None = None,
    ):
        self._compressor = compressor or build_llmlingua2_compressor(
            model_name=model_name,
            device_map=device_map,
        )

    @staticmethod
    def _concat_context(request: ContextRequest) -> str:
        return "\n\n".join(rec.text for rec in request.records)

    def select(self, request: ContextRequest) -> ContextResponse:
        t0 = time.perf_counter()
        context = self._concat_context(request)
        full_context_tokens = estimate_tokens(context)

        # Compress the concatenated context toward the token budget. The
        # compression rate is derived from the budget the same way the
        # reference adapter derives it, clamped to LLMLingua-2's usable range.
        target_tokens = max(1, int(request.budget.max_context_tokens))
        rate = target_tokens / full_context_tokens if full_context_tokens > 0 else 1.0
        rate = min(1.0, max(0.05, rate))

        compression = self._compressor.compress(context, rate=rate)
        compressed_context = compression.get("compressed_prompt") or context

        latency_ms = (time.perf_counter() - t0) * 1000.0

        tokens_before = full_context_tokens
        tokens_after = estimate_tokens(compressed_context)
        tokens_saved = max(0, tokens_before - tokens_after)
        reduction_ratio = (tokens_saved / tokens_before) if tokens_before > 0 else 0.0

        return ContextResponse(
            request_id=request.request_id,
            rendered_context=compressed_context,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            records_available=len(request.records),
            records_selected=len(request.records),
            fallback_used=False,
            engine_latency_ms=latency_ms,
            budget_tokens=request.budget.max_context_tokens,
            reduction_ratio=reduction_ratio,
            safety=None,
            gate=None,
        )
