"""The uniform arm contract.

Every arm, local (in-process) or hosted (over HTTP), accepts a
``ContextRequest`` and returns a ``ContextResponse``. These are plain,
JSON-serializable data types. They carry no method-internal tuning: a hosted
method resolves its own configuration from an opaque, versioned
``operating_point`` label (see ``INTERFACE.md``), so the request never has to
name any method's knobs.

Field names/types are stable and form the wire schema. ``to_wire`` /
``from_wire`` round-trip to and from JSON-friendly dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


@dataclass
class ContextRecord:
    """One unit of state the method may keep, drop, or compress."""

    text: str
    kind: str = "external_data"
    id: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    step_id: Optional[str] = None
    importance: float = 0.0
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "ContextRecord":
        return cls(
            text=d["text"],
            kind=d.get("kind", "external_data"),
            id=d.get("id"),
            source=d.get("source"),
            title=d.get("title"),
            step_id=d.get("step_id"),
            importance=float(d.get("importance", 0.0)),
            keywords=list(d.get("keywords", [])),
            tags=list(d.get("tags", [])),
            attributes=dict(d.get("attributes", {})),
        )


@dataclass
class TaskSpec:
    """The query the selected context must serve."""

    prompt: str
    tool_name: Optional[str] = None
    required_record_ids: list[str] = field(default_factory=list)
    parent_record_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    step_id: Optional[str] = None
    recent_prompts: list[str] = field(default_factory=list)
    output_mode: Optional[str] = None
    output_token_budget: Optional[int] = None

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "TaskSpec":
        return cls(
            prompt=d["prompt"],
            tool_name=d.get("tool_name"),
            required_record_ids=list(d.get("required_record_ids", [])),
            parent_record_ids=list(d.get("parent_record_ids", [])),
            keywords=list(d.get("keywords", [])),
            tags=list(d.get("tags", [])),
            step_id=d.get("step_id"),
            recent_prompts=list(d.get("recent_prompts", [])),
            output_mode=d.get("output_mode"),
            output_token_budget=d.get("output_token_budget"),
        )


@dataclass
class AdaptiveBudget:
    """Budget-escalation ladder for ``mode="adaptive"``."""

    initial_tokens: int
    escalation_tokens: list[int] = field(default_factory=list)
    allow_full_context_fallback: bool = True

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "AdaptiveBudget":
        return cls(
            initial_tokens=int(d["initial_tokens"]),
            escalation_tokens=list(d.get("escalation_tokens", [])),
            allow_full_context_fallback=bool(d.get("allow_full_context_fallback", True)),
        )


@dataclass
class BudgetSpec:
    """The operating point. Arm-neutral quantities only.

    ``operating_point`` is an opaque, versioned label a hosted method maps to
    its own configuration server-side; the fields below are arm-neutral
    overrides that every method honors.
    """

    max_context_tokens: int
    operating_point: Optional[str] = None
    max_records: Optional[int] = None
    max_excerpt_tokens_per_record: Optional[int] = None
    mode: str = "fixed"  # "fixed" | "adaptive"
    adaptive: Optional[AdaptiveBudget] = None
    require_evidence_coverage: Optional[bool] = None

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "BudgetSpec":
        adaptive = d.get("adaptive")
        return cls(
            max_context_tokens=int(d["max_context_tokens"]),
            operating_point=d.get("operating_point"),
            max_records=d.get("max_records"),
            max_excerpt_tokens_per_record=d.get("max_excerpt_tokens_per_record"),
            mode=d.get("mode", "fixed"),
            adaptive=AdaptiveBudget.from_wire(adaptive) if adaptive else None,
            require_evidence_coverage=d.get("require_evidence_coverage"),
        )


@dataclass
class ContextRequest:
    request_id: str
    records: list[ContextRecord]
    task: TaskSpec
    budget: BudgetSpec
    render: bool = True
    render_format: str = "plain"  # "plain" | "hybrid"
    return_per_record: bool = True

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "ContextRequest":
        return cls(
            request_id=d["request_id"],
            records=[ContextRecord.from_wire(r) for r in d.get("records", [])],
            task=TaskSpec.from_wire(d["task"]),
            budget=BudgetSpec.from_wire(d["budget"]),
            render=bool(d.get("render", True)),
            render_format=d.get("render_format", "plain"),
            return_per_record=bool(d.get("return_per_record", True)),
        )


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


@dataclass
class SelectedRecord:
    record_id: str
    kind: str = "external_data"
    title: Optional[str] = None
    source: Optional[str] = None
    score: float = 0.0
    reason: str = ""
    excerpt: str = ""
    excerpt_format: str = "plain"
    selected_tokens: int = 0

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "SelectedRecord":
        return cls(
            record_id=d["record_id"],
            kind=d.get("kind", "external_data"),
            title=d.get("title"),
            source=d.get("source"),
            score=float(d.get("score", 0.0)),
            reason=d.get("reason", ""),
            excerpt=d.get("excerpt", ""),
            excerpt_format=d.get("excerpt_format", "plain"),
            selected_tokens=int(d.get("selected_tokens", 0)),
        )


@dataclass
class SafetySummary:
    """Neutral subset of a method's coverage/answerability verdict.

    Only the outcome crosses the boundary; a method's internal obligation
    taxonomy and repair internals stay server-side.
    """

    selection_safe: bool = True
    fallback_required: bool = False
    fallback_reason: str = ""
    coverage_score: float = 0.0
    evidence_shape: str = "unknown"
    evidence_terms: Optional[dict[str, list[str]]] = None
    repair_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "SafetySummary":
        return cls(
            selection_safe=bool(d.get("selection_safe", True)),
            fallback_required=bool(d.get("fallback_required", False)),
            fallback_reason=d.get("fallback_reason", ""),
            coverage_score=float(d.get("coverage_score", 0.0)),
            evidence_shape=d.get("evidence_shape", "unknown"),
            evidence_terms=d.get("evidence_terms"),
            repair_reasons=list(d.get("repair_reasons", [])),
        )


@dataclass
class GateSummary:
    """Neutral engage/stand-down summary. ``signals`` is JSON-safe telemetry."""

    engaged: bool = False
    reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "GateSummary":
        return cls(
            engaged=bool(d.get("engaged", False)),
            reason=d.get("reason", ""),
            signals=dict(d.get("signals", {})),
        )


@dataclass
class ContextResponse:
    request_id: str
    rendered_context: str
    policy_version: Optional[str] = None
    selected: list[SelectedRecord] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    records_available: int = 0
    records_selected: int = 0
    fallback_used: bool = False
    selection_error: Optional[str] = None
    engine_latency_ms: float = 0.0
    budget_tokens: int = 0
    attempted_budget_tokens: list[int] = field(default_factory=list)
    reduction_ratio: float = 0.0
    safety: Optional[SafetySummary] = None
    gate: Optional[GateSummary] = None
    format_metrics: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "ContextResponse":
        safety = d.get("safety")
        gate = d.get("gate")
        return cls(
            request_id=d["request_id"],
            rendered_context=d.get("rendered_context", ""),
            policy_version=d.get("policy_version"),
            selected=[SelectedRecord.from_wire(s) for s in d.get("selected", [])],
            tokens_before=int(d.get("tokens_before", 0)),
            tokens_after=int(d.get("tokens_after", 0)),
            tokens_saved=int(d.get("tokens_saved", 0)),
            records_available=int(d.get("records_available", 0)),
            records_selected=int(d.get("records_selected", 0)),
            fallback_used=bool(d.get("fallback_used", False)),
            selection_error=d.get("selection_error"),
            engine_latency_ms=float(d.get("engine_latency_ms", 0.0)),
            budget_tokens=int(d.get("budget_tokens", 0)),
            attempted_budget_tokens=list(d.get("attempted_budget_tokens", [])),
            reduction_ratio=float(d.get("reduction_ratio", 0.0)),
            safety=SafetySummary.from_wire(safety) if safety else None,
            gate=GateSummary.from_wire(gate) if gate else None,
            format_metrics=dict(d.get("format_metrics", {})),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Fields every conformant ContextResponse must carry (checked at the HTTP
# boundary by the client and the stub server).
REQUIRED_RESPONSE_FIELDS = (
    "request_id",
    "rendered_context",
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "records_available",
    "records_selected",
    "fallback_used",
    "engine_latency_ms",
)
