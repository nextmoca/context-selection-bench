"""Full-context passthrough arm: the no-reduction control.

Concatenates every record in document order and returns it verbatim. This is
the reference every reduction method is measured against: same records in, all
records out, nothing saved.
"""

from __future__ import annotations

import time

from ..contracts import (
    ContextRequest,
    ContextResponse,
    SelectedRecord,
)
from ..tokenizing import estimate_tokens
from .base import ContextArm


def render_records(request: ContextRequest) -> str:
    """Join record text in order, with a light title header when present."""
    blocks: list[str] = []
    for rec in request.records:
        header = f"[{rec.title}]\n" if rec.title else ""
        blocks.append(f"{header}{rec.text}")
    return "\n\n".join(blocks)


class FullContextArm(ContextArm):
    name = "full_context"

    def select(self, request: ContextRequest) -> ContextResponse:
        t0 = time.perf_counter()
        rendered = render_records(request)
        tokens = estimate_tokens(rendered)
        selected = [
            SelectedRecord(
                record_id=(rec.id or f"rec-{i}"),
                kind=rec.kind,
                title=rec.title,
                source=rec.source,
                score=0.0,
                reason="full_context",
                excerpt=rec.text,
                excerpt_format="plain",
                selected_tokens=estimate_tokens(rec.text),
            )
            for i, rec in enumerate(request.records)
        ]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ContextResponse(
            request_id=request.request_id,
            rendered_context=rendered,
            selected=selected if request.return_per_record else [],
            tokens_before=tokens,
            tokens_after=tokens,
            tokens_saved=0,
            records_available=len(request.records),
            records_selected=len(request.records),
            fallback_used=False,
            engine_latency_ms=latency_ms,
            budget_tokens=request.budget.max_context_tokens,
            reduction_ratio=0.0,
            safety=None,
            gate=None,
        )
