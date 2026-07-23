"""Compresr arm: a thin client for the Compresr context-compression API.

Compresr is a commercial compression service (`pip install compresr`). This arm
sends the request's records as the context and the task prompt as the query to
Compresr's question-specific compression, and returns the compressed context as
the arm's rendered context. It runs at Compresr's default operating point: one
model (`latte_v1`, the only question-specific model), no target ratio (the
service's own default), no tuning, the same one-arm-one-setting discipline every
arm follows.

Token counts in the response use the harness's shared estimator (so all arms are
counted identically); Compresr's own token/ratio figures are preserved under
``format_metrics`` for reference. Requires ``COMPRESR_API_KEY`` in the
environment and the ``compresr`` optional dependency.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from ..contracts import ContextRequest, ContextResponse
from ..latency import MeasurementClass
from ..tokenizing import estimate_tokens
from .base import ContextArm

# Question-specific compression accepts only this model in the published SDK.
QUESTION_SPECIFIC_MODEL = "latte_v1"


def _concat(request: ContextRequest) -> str:
    return "\n\n".join(rec.text for rec in request.records)


class CompresrArm(ContextArm):
    name = "compresr"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = QUESTION_SPECIFIC_MODEL,
        target_compression_ratio: Optional[float] = None,  # None -> service default (no tuning)
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Any = None,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._ratio = target_compression_ratio
        self._base_url = base_url
        self._timeout = timeout
        self._client = client  # injectable for tests
        self._api_key = api_key or os.environ.get("COMPRESR_API_KEY")
        self._max_retries = max_retries

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("COMPRESR_API_KEY is required to run the Compresr arm")
            from compresr import CompressionClient  # lazy: optional 'compresr' extra

            self._client = CompressionClient(
                api_key=self._api_key, base_url=self._base_url, timeout=self._timeout
            )
        return self._client

    def _compress(self, context: str, query: str) -> Any:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "context": context,
            "query": query,
            "compression_model_name": self._model,
        }
        if self._ratio is not None:
            kwargs["target_compression_ratio"] = self._ratio
        # The SDK has no built-in retry; back off on rate limits using Retry-After.
        for attempt in range(self._max_retries):
            try:
                return client.compress(**kwargs)
            except Exception as exc:  # noqa: BLE001 - SDK raises typed errors we don't import
                retry_after = getattr(exc, "retry_after", None)
                if retry_after is None or attempt == self._max_retries - 1:
                    raise
                time.sleep(float(retry_after))
        raise RuntimeError("unreachable")

    def select(self, request: ContextRequest) -> ContextResponse:
        context = _concat(request)
        query = request.task.prompt
        t0 = time.perf_counter()
        result = self._compress(context, query)
        client_latency_ms = (time.perf_counter() - t0) * 1000.0

        data = result.data
        compressed = data.compressed_context
        tokens_before = estimate_tokens(context)
        tokens_after = estimate_tokens(compressed)
        tokens_saved = max(0, tokens_before - tokens_after)

        return ContextResponse(
            request_id=request.request_id,
            rendered_context=compressed,
            selected=[],  # Compresr compresses text; it does not select at record granularity
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            records_available=len(request.records),
            records_selected=len(request.records),
            fallback_used=False,
            engine_latency_ms=float(getattr(data, "duration_ms", 0) or client_latency_ms),
            budget_tokens=request.budget.max_context_tokens,
            reduction_ratio=(tokens_saved / tokens_before if tokens_before else 0.0),
            safety=None,
            gate=None,
            format_metrics={
                "provider": "compresr",
                "compresr_model": self._model,
                "compresr_original_tokens": getattr(data, "original_tokens", None),
                "compresr_compressed_tokens": getattr(data, "compressed_tokens", None),
                "compresr_actual_ratio": getattr(data, "actual_compression_ratio", None),
                "compresr_tokens_saved": getattr(data, "tokens_saved", None),
                # client_latency_ms is the full wall-clock round trip to the
                # remote API (transport included) -> remote_api_e2e. It also
                # includes any rate-limit backoff sleeps, so use the MEDIAN
                # across items, not the mean/p90, as the honest per-item figure.
                "client_latency_ms": client_latency_ms,
                "client_latency_ms_class": MeasurementClass.REMOTE_API_E2E,
                # duration_ms is Compresr's OWN server-side timer (vendor
                # self-reported), not one of our measurement classes; kept as a
                # cross-check only.
                "vendor_reported_duration_ms": getattr(data, "duration_ms", None),
            },
        )
