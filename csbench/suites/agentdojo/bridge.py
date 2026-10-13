"""The selection boundary: a pipeline element inserted after the tool executor.

This element is a transport. It collects the contiguous trailing tool-result
messages, builds a raw text candidate, submits it through the hosted selector's
public simple-text ingress (``text`` + ``budget.autochunk``), and applies the
rendered selection the server returned when, and only when, the server
engaged. It never decides what to keep: no chunking, ranking, scoring or
filtering happens here. On every other outcome (stand-down, fallback, empty or
malformed response, timeout, exception) the original tool messages are passed
to the model unchanged, byte for byte.

Only the trailing tool-result batch is ever submitted. System, user, assistant,
developer, policy and tool-schema content is never part of a request.

The element works against AgentDojo's plain-dict message shape (``role``,
``content`` as a list of ``{"type": "text", "content": str}`` blocks, and for
tool messages ``tool_call``, ``tool_call_id``, ``error``). When ``agentdojo`` is
installed it subclasses ``BasePipelineElement``; otherwise a minimal shim with
the same ``query`` signature is used so the module (and its tests) work offline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .config import BudgetPolicy, SelectorIdentity
from .ids import case_id, turn_request_id
from .telemetry import (
    EVENT_APPLIED,
    EVENT_FAIL_OPEN,
    EVENT_PASSTHROUGH,
    INGRESS_TEXT,
    SelectionEvent,
    TelemetryWriter,
)

try:  # pragma: no cover - exercised only when agentdojo is installed
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement as _Base

    AGENTDOJO_AVAILABLE = True
except Exception:  # ImportError or a transitive import failure
    AGENTDOJO_AVAILABLE = False

    class _Base:  # type: ignore[no-redef]
        """Shim with AgentDojo's element signature (used when agentdojo is absent)."""

        name: Optional[str] = None

        def query(self, query, runtime, env=None, messages=(), extra_args=None):  # noqa: D401
            raise NotImplementedError


# Fields the public simple-text response must carry for the bridge to read it.
# (The same set the hosted-adapter client validates in ``csbench.arms``.)
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

OMISSION_MARKER = "[tool result omitted: its selected content was delivered in tool result 1]"
TASK_TOOL_NAME = "agentdojo"
TASK_TAGS = ("agentdojo", "tool_result_selection")


def protocol_token_estimate(text: str) -> int:
    """The protocol's deterministic estimator (not provider-billed accounting)."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Message helpers (pure)
# --------------------------------------------------------------------------- #


def trailing_tool_batch_start(messages: Sequence[dict[str, Any]]) -> Optional[int]:
    """Index where the contiguous trailing run of ``role == "tool"`` messages
    starts, or ``None`` when the newest message is not a tool result."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    i = len(messages) - 1
    while i > 0 and messages[i - 1].get("role") == "tool":
        i -= 1
    return i


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("content", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text"
    )


def build_candidate_text(texts: Sequence[str]) -> str:
    """One result is submitted unchanged; several are joined with stable labels."""
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(f"tool result {i}:\n{t}" for i, t in enumerate(texts, start=1))


def build_text_request(
    *,
    request_id: str,
    text: str,
    task_prompt: str,
    budget: BudgetPolicy,
    operating_point: str,
) -> dict[str, Any]:
    """The protocol's logical request, on the public simple-text wire shape."""
    return {
        "request_id": request_id,
        "text": text,
        "task": {
            "prompt": task_prompt,
            "tool_name": TASK_TOOL_NAME,
            "tags": list(TASK_TAGS),
        },
        "budget": {
            "max_context_tokens": int(budget.max_context_tokens),
            "capacity_cap_tokens": int(budget.capacity_cap_tokens),
            "operating_point": operating_point,
            "max_records": int(budget.max_records),
            "mode": budget.mode,
            "autochunk": bool(budget.autochunk),
        },
        "render": True,
        "render_format": "plain",
        "return_per_record": True,
    }


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "content": text}


def apply_selection(tool_messages: Sequence[dict[str, Any]], rendered: str) -> list[dict[str, Any]]:
    """Rendered context replaces the first tool result; the rest become markers.

    Message count, order, ``tool_call``, ``tool_call_id`` and ``error`` are
    preserved so the model adapter sees the structure it expects.
    """
    out: list[dict[str, Any]] = []
    for i, m in enumerate(tool_messages):
        new = dict(m)
        new["content"] = [_text_block(rendered if i == 0 else OMISSION_MARKER)]
        out.append(new)
    return out


# --------------------------------------------------------------------------- #
# Response classification (pure)
# --------------------------------------------------------------------------- #


@dataclass
class Decision:
    event: str
    reason: str
    rendered: Optional[str] = None
    gate_reason: Optional[str] = None
    engine_latency_ms: Optional[float] = None
    support_contract: Optional[str] = None
    semantic_adapters: Optional[list[str]] = None
    records_expanded: Optional[int] = None
    derived_records: Optional[int] = None

    @property
    def applied(self) -> bool:
        return self.event == EVENT_APPLIED


def _opt_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _opt_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _routing(response: dict[str, Any]) -> dict[str, Any]:
    """Routing identity, read from wherever the response carries it (gate
    signals or format metrics); absent fields stay ``None``."""
    gate = response.get("gate") if isinstance(response.get("gate"), dict) else {}
    signals = gate.get("signals") if isinstance(gate.get("signals"), dict) else {}
    fm = response.get("format_metrics") if isinstance(response.get("format_metrics"), dict) else {}
    sources = (response, signals, fm)

    def pick(key: str) -> Any:
        for src in sources:
            if key in src:
                return src[key]
        return None

    adapters = pick("semantic_adapters")
    if isinstance(adapters, str):
        adapters = [adapters]
    elif isinstance(adapters, list):
        adapters = [str(a) for a in adapters]
    else:
        adapters = None
    return {
        "gate_reason": _opt_str(gate.get("reason")) if gate else None,
        "support_contract": _opt_str(pick("support_contract")),
        "semantic_adapters": adapters,
        "records_expanded": _opt_int(pick("records_expanded")),
        "derived_records": _opt_int(pick("derived_records")),
    }


def classify_response(request_id: str, response: Any) -> Decision:
    """Decide applied / passthrough / fail-open from the server's answer.

    The decision reads only outcome flags the server set; it never inspects
    the content to choose what to keep. A selection is applied only on an
    explicit ``outcome == "engaged"`` with non-empty rendered text and no
    fallback, safety or gate signal to the contrary.
    """
    if not isinstance(response, dict):
        return Decision(EVENT_FAIL_OPEN, "malformed_response:not_an_object")
    missing = [f for f in REQUIRED_RESPONSE_FIELDS if f not in response]
    if missing:
        return Decision(EVENT_FAIL_OPEN, "malformed_response:missing_fields")
    if response.get("request_id") != request_id:
        return Decision(EVENT_FAIL_OPEN, "malformed_response:request_id_mismatch")
    rendered = response.get("rendered_context")
    if not isinstance(rendered, str):
        return Decision(EVENT_FAIL_OPEN, "malformed_response:rendered_context_not_text")

    routing = _routing(response)
    engine_latency = _opt_float(response.get("engine_latency_ms"))
    common = dict(engine_latency_ms=engine_latency, **routing)

    outcome = response.get("outcome")
    safety = response.get("safety") if isinstance(response.get("safety"), dict) else {}
    if outcome == "stood_down":
        return Decision(EVENT_PASSTHROUGH, "stood_down", **common)
    if outcome == "escalated":
        return Decision(EVENT_PASSTHROUGH, "escalated", **common)
    if response.get("fallback_used") is True:
        return Decision(EVENT_PASSTHROUGH, "fallback_used", **common)
    if safety.get("fallback_required") is True or safety.get("selection_safe") is False:
        return Decision(EVENT_PASSTHROUGH, "unsafe_selection", **common)
    if response.get("selection_error"):
        return Decision(EVENT_PASSTHROUGH, "selection_error", **common)
    # Application requires explicit engagement. A missing outcome is not an
    # engagement, and an outcome this bridge does not know (open enum) is not
    # one either; a gate that says it did not engage overrides everything.
    if outcome is None:
        return Decision(EVENT_PASSTHROUGH, "no_outcome", **common)
    if outcome != "engaged":
        return Decision(EVENT_PASSTHROUGH, "unknown_outcome", **common)
    gate = response.get("gate")
    if isinstance(gate, dict) and gate.get("engaged") is False:
        return Decision(EVENT_PASSTHROUGH, "gate_not_engaged", **common)
    if not rendered.strip():
        return Decision(EVENT_FAIL_OPEN, "empty_selection", **common)
    return Decision(EVENT_APPLIED, "applied", rendered=rendered, **common)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


class SelectorTimeout(RuntimeError):
    """The selector did not answer within the configured timeout."""


class HttpTextTransport:
    """POST the request to the public endpoint; return the parsed JSON body.

    The API key is read from the environment at call time (never stored) and
    sent only as a bearer header. Non-2xx responses raise.
    """

    def __init__(
        self,
        selector: SelectorIdentity,
        *,
        session: Any = None,
        environ: Optional[dict[str, str]] = None,
    ) -> None:
        if not selector.base_url:
            raise ValueError("selector.base_url is required for the HTTP transport")
        self.url = selector.base_url.rstrip("/") + selector.endpoint_path
        self.timeout_s = float(selector.timeout_s)
        self.key_env = tuple(selector.api_key_env)
        self._environ = os.environ if environ is None else environ
        self._session = session

    def _api_key(self) -> Optional[str]:
        for name in self.key_env:
            value = self._environ.get(name)
            if value:
                return value
        return None

    def __call__(self, request: dict[str, Any]) -> Any:
        import requests

        if self._session is None:
            self._session = requests.Session()  # kept: one connection pool per shard process
        session = self._session
        headers = {"content-type": "application/json"}
        key = self._api_key()
        if key:
            headers["authorization"] = f"Bearer {key}"
        try:
            resp = session.post(self.url, json=request, headers=headers, timeout=self.timeout_s)
        except requests.Timeout as exc:
            raise SelectorTimeout(str(type(exc).__name__)) from None
        resp.raise_for_status()
        return resp.json()


# --------------------------------------------------------------------------- #
# Case context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseContext:
    user_task_id: str
    injection_task_id: str


def agentdojo_case_context() -> CaseContext:
    """Read the current case from AgentDojo's active trace logger."""
    from agentdojo.logging import Logger

    logger = Logger.get()
    ctx = getattr(logger, "context", {}) or {}
    return CaseContext(
        user_task_id=str(ctx.get("user_task_id", "unknown_user_task")),
        injection_task_id=str(ctx.get("injection_task_id") or "none"),
    )


# --------------------------------------------------------------------------- #
# The element
# --------------------------------------------------------------------------- #


@dataclass
class BridgeIdentity:
    suite: str
    attack: str
    repeat: int
    arm: str


class NeedlepathToolResultBridge(_Base):
    """Pipeline element: public-text selection over the trailing tool batch."""

    name = None  # the pipeline keeps the LLM's name; this element adds none

    def __init__(
        self,
        *,
        identity: BridgeIdentity,
        selector: SelectorIdentity,
        budget: BudgetPolicy,
        transport: Callable[[dict[str, Any]], Any],
        telemetry: Optional[TelemetryWriter] = None,
        case_context: Callable[[], CaseContext] = agentdojo_case_context,
        token_estimate: Callable[[str], int] = protocol_token_estimate,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.identity = identity
        self.selector = selector
        self.budget = budget
        self.transport = transport
        self.telemetry = telemetry
        self.case_context = case_context
        self.token_estimate = token_estimate
        self.clock = clock
        self._turns: dict[tuple[str, str], int] = {}
        self.events: list[SelectionEvent] = []  # in-memory copy (tests, per-case join)

    # -- turn bookkeeping -------------------------------------------------- #
    #
    # The turn index counts submitted batches per case for the lifetime of this
    # element (one shard process). AgentDojo may re-query a case in-process when
    # the model produced no output; the counter keeps climbing rather than
    # restarting, so every request id a process emits is unique.

    def _next_turn(self, ctx: CaseContext) -> int:
        key = (ctx.user_task_id, ctx.injection_task_id)
        n = self._turns.get(key, 0)
        self._turns[key] = n + 1
        return n

    # -- the hop ------------------------------------------------------------ #

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        start = trailing_tool_batch_start(messages)
        if start is None:
            return query, runtime, env, messages, extra_args

        tool_messages = list(messages[start:])
        texts = [message_text(m) for m in tool_messages]
        candidate = build_candidate_text(texts)

        ctx = self.case_context()
        turn = self._next_turn(ctx)
        ident = self.identity
        cid = case_id(ident.suite, ident.attack, ctx.user_task_id, ctx.injection_task_id, ident.repeat, ident.arm)
        rid = turn_request_id(
            ident.suite, ident.attack, ctx.user_task_id, ctx.injection_task_id, ident.repeat, ident.arm, turn
        )
        request = build_text_request(
            request_id=rid,
            text=candidate,
            task_prompt=query,
            budget=self.budget,
            operating_point=self.selector.operating_point,
        )

        t0 = self.clock()
        try:
            response = self.transport(request)
            decision = classify_response(rid, response)
        except SelectorTimeout:
            decision = Decision(EVENT_FAIL_OPEN, "timeout")
        except Exception as exc:  # fail open on anything the transport raises
            decision = Decision(EVENT_FAIL_OPEN, f"exception:{type(exc).__name__}")
        latency_ms = (self.clock() - t0) * 1000.0

        original_tokens = self.token_estimate(candidate)
        selected_tokens = self.token_estimate(decision.rendered) if decision.applied else original_tokens
        event = SelectionEvent(
            event=decision.event,
            ingress=INGRESS_TEXT,
            autochunk=bool(self.budget.autochunk),
            operating_point=self.selector.operating_point,
            tool_results=len(tool_messages),
            original_token_estimate=original_tokens,
            selected_token_estimate=selected_tokens,
            latency_ms=round(latency_ms, 3),
            engine_latency_ms=decision.engine_latency_ms,
            gate_reason=decision.gate_reason,
            reason=decision.reason,
            support_contract=decision.support_contract,
            semantic_adapters=decision.semantic_adapters,
            records_expanded=decision.records_expanded,
            derived_records=decision.derived_records,
            case_id=cid,
            turn_index=turn,
            request_id=rid,
        )
        self.events.append(event)
        if self.telemetry is not None:
            self.telemetry.write(event)

        if not decision.applied:
            # Preserve the original objects: nothing is copied or re-encoded.
            return query, runtime, env, messages, extra_args

        replaced = apply_selection(tool_messages, decision.rendered or "")
        return query, runtime, env, [*messages[:start], *replaced], extra_args


# --------------------------------------------------------------------------- #
# Pipeline insertion
# --------------------------------------------------------------------------- #


class PipelineShapeError(RuntimeError):
    """The pipeline does not have the tool-execution loop the bridge expects."""


def _is_tools_executor(element: Any) -> bool:
    return type(element).__name__ == "ToolsExecutor"


def insert_after_tool_executor(pipeline: Any, element: Any) -> Any:
    """Insert ``element`` immediately after the tool executor inside the
    pipeline's tool-execution loop. Exactly one executor must exist."""
    hits = 0
    for top in list(getattr(pipeline, "elements", [])):
        inner = getattr(top, "elements", None)
        if inner is None:
            continue
        inner = list(inner)
        for i, el in enumerate(inner):
            if _is_tools_executor(el):
                inner.insert(i + 1, element)
                top.elements = inner
                hits += 1
                break
    if hits != 1:
        raise PipelineShapeError(f"expected exactly one ToolsExecutor inside a tool loop, found {hits}")
    return pipeline
