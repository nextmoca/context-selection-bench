"""CompresrArm mapping + retry, exercised with an injected fake client (no network)."""

from types import SimpleNamespace

import pytest

from csbench.arms.compresr import CompresrArm
from csbench.contracts import BudgetSpec, ContextRecord, ContextRequest, TaskSpec


def _request(text: str = "Python was created in 1991. JavaScript in 1995. Java in 1995.") -> ContextRequest:
    return ContextRequest(
        request_id="req-1",
        records=[ContextRecord(text=text)],
        task=TaskSpec(prompt="Who created Python?"),
        budget=BudgetSpec(max_context_tokens=4096),
    )


class _FakeClient:
    def __init__(self, compressed="Python created 1991.", **counts):
        self._compressed = compressed
        self._counts = counts
        self.calls = []

    def compress(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            data=SimpleNamespace(
                original_context=kwargs["context"],
                compressed_context=self._compressed,
                original_tokens=self._counts.get("original_tokens", 20),
                compressed_tokens=self._counts.get("compressed_tokens", 4),
                actual_compression_ratio=self._counts.get("actual_compression_ratio", 0.8),
                tokens_saved=self._counts.get("tokens_saved", 16),
                duration_ms=self._counts.get("duration_ms", 42),
            ),
        )


def test_compresr_arm_maps_response():
    fake = _FakeClient(compressed="Python created 1991.")
    arm = CompresrArm(client=fake)
    resp = arm.select(_request())

    assert resp.request_id == "req-1"
    assert resp.rendered_context == "Python created 1991."
    assert resp.records_available == 1
    assert resp.fallback_used is False
    assert resp.tokens_after < resp.tokens_before  # it compressed
    assert resp.tokens_saved == resp.tokens_before - resp.tokens_after
    assert resp.engine_latency_ms == 42.0  # server duration_ms
    # native Compresr figures preserved for reference
    assert resp.format_metrics["provider"] == "compresr"
    assert resp.format_metrics["compresr_actual_ratio"] == 0.8
    # the call used the question-specific model + query
    assert fake.calls[0]["compression_model_name"] == "latte_v1"
    assert fake.calls[0]["query"] == "Who created Python?"
    # no tuning: target ratio omitted
    assert "target_compression_ratio" not in fake.calls[0]


def test_compresr_arm_retries_on_rate_limit():
    class _RateLimited:
        def __init__(self):
            self.n = 0

        def compress(self, **kwargs):
            self.n += 1
            if self.n == 1:
                err = RuntimeError("429")
                err.retry_after = 0  # no real sleep
                raise err
            return SimpleNamespace(
                success=True,
                data=SimpleNamespace(compressed_context="ok", original_tokens=5, compressed_tokens=1, actual_compression_ratio=0.8, tokens_saved=4, duration_ms=1),
            )

    arm = CompresrArm(client=_RateLimited())
    resp = arm.select(_request())
    assert resp.rendered_context == "ok"


def test_compresr_arm_requires_key_without_client(monkeypatch):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    arm = CompresrArm(api_key=None)  # no injected client, no env key path
    with pytest.raises(RuntimeError, match="COMPRESR_API_KEY"):
        arm.select(_request())
