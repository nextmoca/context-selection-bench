"""Bridge regression tests: baseline equivalence, applied selection, fail-open,
context isolation, public-text contract, telemetry privacy. All offline."""

from __future__ import annotations

import json

import pytest

from agentdojo_fakes import (
    RecordingTransport,
    conversation_with_tool_batch,
    fixed_case_context,
    freeze,
    selector_response,
)
from csbench.suites.agentdojo import ARM_NEEDLEPATH
from csbench.suites.agentdojo.bridge import (
    OMISSION_MARKER,
    BridgeIdentity,
    NeedlepathToolResultBridge,
    PipelineShapeError,
    SelectorTimeout,
    build_candidate_text,
    decide_response,
    insert_after_tool_executor,
    protocol_token_estimate,
    trailing_tool_batch_start,
)
from csbench.suites.agentdojo.config import BudgetPolicy, SelectorIdentity
from csbench.suites.agentdojo.ids import turn_request_id
from csbench.suites.agentdojo.telemetry import (
    EVENT_APPLIED,
    EVENT_FAIL_OPEN,
    EVENT_FIELDS,
    EVENT_PASSTHROUGH,
    TelemetryWriter,
    read_events,
)

SELECTOR = SelectorIdentity(base_url="http://127.0.0.1:9", snapshot="snap", operating_point="np-2026-10-r6")
BUDGET = BudgetPolicy()
IDENTITY = BridgeIdentity(suite="workspace", attack="important_instructions", repeat=1, arm=ARM_NEEDLEPATH)
SECRET = "fake-key-value-never-logged-0123456789"


def make_bridge(responder, tmp_path=None, **kw):
    transport = RecordingTransport(responder)
    telemetry = TelemetryWriter(tmp_path / "needlepath_metrics_shard_0.jsonl") if tmp_path else None
    bridge = NeedlepathToolResultBridge(
        identity=IDENTITY,
        selector=SELECTOR,
        budget=BUDGET,
        transport=transport,
        telemetry=telemetry,
        case_context=fixed_case_context(),
        **kw,
    )
    return bridge, transport


def run(bridge, messages):
    _, _, _, out, _ = bridge.query("USER TASK: find the invoice total", None, None, messages, {})
    return out


# ----------------------------------------------------------------- helpers


def test_trailing_batch_detection():
    msgs = conversation_with_tool_batch(["a", "b", "c"])
    assert trailing_tool_batch_start(msgs) == 3
    assert trailing_tool_batch_start(msgs[:3]) is None  # newest is the assistant turn
    assert trailing_tool_batch_start([]) is None


def test_candidate_text_single_unchanged_multi_labelled():
    assert build_candidate_text(["only one"]) == "only one"
    assert build_candidate_text(["x", "y"]) == "tool result 1:\nx\n\ntool result 2:\ny"


def test_protocol_token_estimate():
    assert protocol_token_estimate("") == 1
    assert protocol_token_estimate("a" * 400) == 100


# ------------------------------------------------- baseline equivalence


def test_baseline_equivalence_without_bridge_is_identity():
    """A pipeline with the bridge disabled never touches the message flow: the
    tool messages the executor produced are exactly what the model receives."""
    msgs = conversation_with_tool_batch(["TOOL OUTPUT A", "TOOL OUTPUT B"])
    before = freeze(msgs)

    class Executor:  # stands in for ToolsExecutor on the baseline arm
        def query(self, q, r, e, m, x):
            return q, r, e, m, x

    q, r, e, out, x = Executor().query("task", None, None, msgs, {})
    assert out is msgs and freeze(out) == before


def test_passthrough_returns_the_same_message_objects(tmp_path):
    msgs = conversation_with_tool_batch(["TOOL OUTPUT A", "TOOL OUTPUT B"])
    before = freeze(msgs)
    bridge, _ = make_bridge(lambda req: selector_response(req, rendered="", outcome="stood_down"), tmp_path)
    out = run(bridge, msgs)
    assert out is msgs, "passthrough must hand back the original list, not a copy"
    assert freeze(out) == before


# ------------------------------------------------------ applied selection


def test_applied_selection_replaces_only_the_tool_batch(tmp_path):
    msgs = conversation_with_tool_batch(["TOOL OUTPUT A", "TOOL OUTPUT B", "TOOL OUTPUT C"])
    head_before = freeze(msgs[:3])
    bridge, transport = make_bridge(lambda req: selector_response(req, rendered="SELECTED A"), tmp_path)
    out = run(bridge, msgs)

    assert len(out) == len(msgs), "message count is preserved"
    assert freeze(out[:3]) == head_before, "system/user/assistant untouched"
    tools = out[3:]
    assert [m["role"] for m in tools] == ["tool"] * 3
    assert tools[0]["content"] == [{"type": "text", "content": "SELECTED A"}]
    assert tools[1]["content"] == [{"type": "text", "content": OMISSION_MARKER}]
    assert tools[2]["content"] == [{"type": "text", "content": OMISSION_MARKER}]
    # tool identity preserved so the model adapter can pair results with calls
    for before, after in zip(msgs[3:], tools):
        assert after["tool_call"] is before["tool_call"]
        assert after["tool_call_id"] == before["tool_call_id"]
        assert after["error"] == before["error"]
    ev = bridge.events[-1]
    assert ev.event == EVENT_APPLIED and ev.tool_results == 3
    assert ev.selected_token_estimate == protocol_token_estimate("SELECTED A")
    assert ev.original_token_estimate == protocol_token_estimate(transport.requests[0]["text"])


def test_applied_single_result_has_no_marker(tmp_path):
    msgs = conversation_with_tool_batch(["ONE LONG TOOL OUTPUT"])
    bridge, _ = make_bridge(lambda req: selector_response(req, rendered="SHORT"), tmp_path)
    out = run(bridge, msgs)
    assert out[-1]["content"] == [{"type": "text", "content": "SHORT"}]
    assert OMISSION_MARKER not in freeze(out)


def test_no_tool_batch_means_no_request(tmp_path):
    msgs = conversation_with_tool_batch(["x"])[:3]  # ends with the assistant turn
    bridge, transport = make_bridge(lambda req: selector_response(req, rendered="y"), tmp_path)
    out = run(bridge, msgs)
    assert out is msgs and transport.requests == [] and bridge.events == []


# -------------------------------------------------------------- fail-open


@pytest.mark.parametrize(
    "name, responder, expected_event, expected_reason",
    [
        ("timeout", lambda req: (_ for _ in ()).throw(SelectorTimeout("ReadTimeout")), EVENT_FAIL_OPEN, "timeout"),
        ("exception", lambda req: (_ for _ in ()).throw(ConnectionError("dns")), EVENT_FAIL_OPEN, "exception:ConnectionError"),
        ("empty", lambda req: selector_response(req, rendered="   \n"), EVENT_FAIL_OPEN, "empty_selection"),
        ("stand_down", lambda req: selector_response(req, rendered="", outcome="stood_down"), EVENT_PASSTHROUGH, "stood_down"),
        ("fallback_flag", lambda req: selector_response(req, rendered="X", fallback_used=True), EVENT_PASSTHROUGH, "fallback_used"),
        ("escalated", lambda req: selector_response(req, rendered="", outcome="escalated"), EVENT_PASSTHROUGH, "escalated"),
        ("unsafe", lambda req: selector_response(req, rendered="X", safety={"selection_safe": False, "fallback_required": True}), EVENT_PASSTHROUGH, "unsafe_selection"),
        ("malformed_not_object", lambda req: "not json object", EVENT_FAIL_OPEN, "malformed_response:not_an_object"),
        ("malformed_missing", lambda req: {"request_id": req["request_id"]}, EVENT_FAIL_OPEN, "malformed_response:missing_fields"),
        ("malformed_id", lambda req: selector_response(req, rendered="X", request_id="other"), EVENT_FAIL_OPEN, "malformed_response:request_id_mismatch"),
        ("malformed_rendered", lambda req: selector_response(req, rendered=None), EVENT_FAIL_OPEN, "malformed_response:rendered_context_not_text"),
        ("unknown_outcome", lambda req: selector_response(req, rendered="X", outcome="something_new", fallback_used=False, safety={"selection_safe": True, "fallback_required": False}), EVENT_PASSTHROUGH, "unknown_outcome"),
        ("no_outcome", lambda req: selector_response(req, rendered="X", outcome=None, fallback_used=False, safety={"selection_safe": True, "fallback_required": False}, gate={"engaged": True, "reason": "engage:fake", "signals": {}}), EVENT_PASSTHROUGH, "no_outcome"),
        ("gate_not_engaged", lambda req: selector_response(req, rendered="X", gate={"engaged": False, "reason": "standdown:fake", "signals": {}}), EVENT_PASSTHROUGH, "gate_not_engaged"),
    ],
)
def test_fail_open_preserves_exact_original_tool_messages(tmp_path, name, responder, expected_event, expected_reason):
    msgs = conversation_with_tool_batch(["TOOL OUTPUT A", "TOOL OUTPUT B"])
    before = freeze(msgs)
    bridge, _ = make_bridge(responder, tmp_path)
    out = run(bridge, msgs)
    assert out is msgs and freeze(out) == before, f"{name}: originals must be preserved byte for byte"
    ev = bridge.events[-1]
    assert (ev.event, ev.reason) == (expected_event, expected_reason)
    assert ev.selected_token_estimate == ev.original_token_estimate, "nothing delivered means nothing saved"


# ------------------------------------------------ mandatory context isolation


def test_context_isolation_only_tool_results_are_submitted(tmp_path):
    msgs = [
        {"role": "system", "content": [{"type": "text", "content": "SYSTEM-SECRET-POLICY"}]},
        {"role": "developer", "content": [{"type": "text", "content": "DEVELOPER-NOTE"}]},
        {"role": "user", "content": [{"type": "text", "content": "USER-INSTRUCTION"}]},
        {"role": "assistant", "content": [{"type": "text", "content": "ASSISTANT-PRIOR"}], "tool_calls": None},
        {"role": "tool", "content": [{"type": "text", "content": "EARLIER-TOOL-RESULT"}], "tool_call_id": "c0", "tool_call": None, "error": None},
        {"role": "assistant", "content": None, "tool_calls": ["call"]},
        {"role": "tool", "content": [{"type": "text", "content": "NEW-TOOL-RESULT-1"}], "tool_call_id": "c1", "tool_call": None, "error": None},
        {"role": "tool", "content": [{"type": "text", "content": "NEW-TOOL-RESULT-2"}], "tool_call_id": "c2", "tool_call": None, "error": None},
    ]
    bridge, transport = make_bridge(lambda req: selector_response(req, rendered="SEL"), tmp_path)
    run(bridge, msgs)
    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req["text"] == "tool result 1:\nNEW-TOOL-RESULT-1\n\ntool result 2:\nNEW-TOOL-RESULT-2"
    wire = json.dumps(req)
    for banned in ("SYSTEM-SECRET-POLICY", "DEVELOPER-NOTE", "USER-INSTRUCTION", "ASSISTANT-PRIOR", "EARLIER-TOOL-RESULT"):
        assert banned not in wire, f"{banned} crossed the selection boundary"
    assert "records" not in req, "no client-side records: the public text shape only"
    assert "schema" not in wire.lower() or "tool_schema" not in wire


# ------------------------------------------------------ public-text contract


def test_public_text_contract_request_shape(tmp_path):
    msgs = conversation_with_tool_batch(["TOOL OUTPUT"])
    bridge, transport = make_bridge(lambda req: selector_response(req, rendered="SEL"), tmp_path)
    run(bridge, msgs)
    req = transport.requests[0]
    assert set(req) == {"request_id", "text", "task", "budget", "render", "render_format", "return_per_record"}
    assert req["task"] == {"prompt": "USER TASK: find the invoice total", "tool_name": "agentdojo", "tags": ["agentdojo", "tool_result_selection"]}
    assert req["budget"] == {
        "max_context_tokens": 1200,
        "capacity_cap_tokens": 120000,
        "operating_point": "np-2026-10-r6",
        "max_records": 8,
        "mode": "fixed",
        "autochunk": True,
    }
    assert req["render"] is True and req["render_format"] == "plain" and req["return_per_record"] is True
    assert req["request_id"] == turn_request_id("workspace", "important_instructions", "user_task_0", "injection_task_0", 1, ARM_NEEDLEPATH, 0)
    ev = bridge.events[0]
    assert ev.ingress == "text" and ev.autochunk is True and ev.operating_point == "np-2026-10-r6"
    assert ev.records_expanded == 2
    assert ev.gate_reason == "engage:fake" and ev.engine_latency_ms == 12.5
    # the documented free-form object passes through by shape only: scalars survive, nesting is dropped
    written = read_events(tmp_path / "needlepath_metrics_shard_0.jsonl")[0]
    assert written["format_metrics"] == {"fake_label": "value_a", "fake_count": 3, "fake_flag": True}


def test_public_text_contract_response_shape_is_the_adapter_boundary():
    """The bridge reads the same required fields the hosted-adapter client validates."""
    from csbench.contracts import REQUIRED_RESPONSE_FIELDS as ADAPTER_FIELDS
    from csbench.suites.agentdojo.bridge import REQUIRED_RESPONSE_FIELDS as BRIDGE_FIELDS

    assert set(BRIDGE_FIELDS) == set(ADAPTER_FIELDS)
    req = {"request_id": "r", "text": "t"}
    assert decide_response("r", selector_response(req, rendered="X")).applied
    stripped = selector_response(req, rendered="X")
    del stripped["tokens_saved"]
    assert decide_response("r", stripped).reason == "malformed_response:missing_fields"


def test_request_ids_are_deterministic_per_case_and_turn(tmp_path):
    msgs = conversation_with_tool_batch(["A"])
    bridge, transport = make_bridge(lambda req: selector_response(req, rendered="S"), tmp_path)
    run(bridge, msgs)
    run(bridge, msgs)
    ids = [r["request_id"] for r in transport.requests]
    assert ids[0].endswith("/t0") and ids[1].endswith("/t1")
    assert ids[0] == turn_request_id("workspace", "important_instructions", "user_task_0", "injection_task_0", 1, ARM_NEEDLEPATH, 0)
    # a second bridge over the same case reproduces the same ids
    bridge2, transport2 = make_bridge(lambda req: selector_response(req, rendered="S"))
    run(bridge2, msgs)
    assert transport2.requests[0]["request_id"] == ids[0]


# ---------------------------------------------------------- telemetry privacy


def test_telemetry_privacy_no_text_or_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("NEEDLEPATH_API_KEY", SECRET)
    tool_text = "TOOL-TEXT-" + "z" * 300
    rendered = "RENDERED-CONTEXT-" + "y" * 300
    msgs = conversation_with_tool_batch([tool_text, "SECOND-TOOL-TEXT"])
    bridge, _ = make_bridge(lambda req: selector_response(req, rendered=rendered), tmp_path)
    run(bridge, msgs)
    bridge2, _ = make_bridge(lambda req: (_ for _ in ()).throw(RuntimeError(SECRET)), tmp_path)
    run(bridge2, msgs)

    raw = (tmp_path / "needlepath_metrics_shard_0.jsonl").read_text()
    for banned in ("TOOL-TEXT-", "SECOND-TOOL-TEXT", "RENDERED-CONTEXT-", "USER TASK", "SYSTEM PROMPT", SECRET):
        assert banned not in raw
    events = read_events(tmp_path / "needlepath_metrics_shard_0.jsonl")
    assert len(events) == 2
    for e in events:
        assert tuple(sorted(e)) == tuple(sorted(EVENT_FIELDS))
    assert events[1]["event"] == EVENT_FAIL_OPEN and events[1]["reason"] == "exception:RuntimeError"


def _event(**over):
    from csbench.suites.agentdojo.telemetry import SelectionEvent

    base = dict(
        event=EVENT_APPLIED, ingress="text", autochunk=True, operating_point="op", tool_results=1,
        original_token_estimate=1, selected_token_estimate=1, latency_ms=1.0, engine_latency_ms=None,
        gate_reason="engage:ok", reason="applied", records_expanded=None, format_metrics={},
        case_id="s/a/u/i/r1/arm", turn_index=0, request_id="s/a/u/i/r1/arm/t0",
    )
    base.update(over)
    return SelectionEvent(**base)


def test_server_controlled_strings_are_sanitized_before_writing(tmp_path):
    """A server could put anything in gate.reason or the free-form diagnostics
    object; only label-shaped tokens reach the file. A label field becomes
    'unlabelled'; a diagnostics entry that is not a scalar label is dropped."""
    from csbench.suites.agentdojo.telemetry import MAX_PASSTHROUGH_KEYS, UNLABELLED

    leaked = "please ignore previous instructions and " + "x" * 300
    w = TelemetryWriter(tmp_path / "t.jsonl")
    too_many = {f"zz_overflow_{i:03d}": i for i in range(MAX_PASSTHROUGH_KEYS + 10)}  # sort after "ok"
    w.write(_event(gate_reason=leaked, format_metrics={"free_text": leaked, "bad key!": "v", "ok": "fine", "n": 2.5, "nested": [1], **too_many}))
    raw = (tmp_path / "t.jsonl").read_text()
    assert "ignore previous" not in raw and "bad key" not in raw and "nested" not in raw
    ev = read_events(tmp_path / "t.jsonl")[0]
    assert ev["gate_reason"] == UNLABELLED
    assert ev["format_metrics"]["ok"] == "fine" and ev["format_metrics"]["n"] == 2.5
    assert "free_text" not in ev["format_metrics"] and len(ev["format_metrics"]) <= MAX_PASSTHROUGH_KEYS


def test_non_conforming_telemetry_file_refuses_to_load(tmp_path):
    from csbench.suites.agentdojo.telemetry import TelemetryContractError, assert_event_is_metadata_only

    wire = _event().to_wire()
    wire["gate_reason"] = "free text with spaces"
    with pytest.raises(TelemetryContractError, match="gate_reason"):
        assert_event_is_metadata_only(wire)
    (tmp_path / "bad.jsonl").write_text(json.dumps(wire) + "\n")
    with pytest.raises(TelemetryContractError):
        read_events(tmp_path / "bad.jsonl")
    wire["gate_reason"] = "engage:ok"
    wire["format_metrics"] = {"nested": {"x": 1}}
    with pytest.raises(TelemetryContractError, match="scalar label"):
        assert_event_is_metadata_only(wire)
    wire["format_metrics"] = {}
    wire["extra_field"] = 1
    with pytest.raises(TelemetryContractError, match="drifted"):
        assert_event_is_metadata_only(wire)


# ------------------------------------------------------- pipeline insertion


class _Loop:
    def __init__(self, elements):
        self.elements = elements


class _Pipeline:
    def __init__(self, elements):
        self.elements = elements


def test_insert_after_tool_executor_with_fake_shapes():
    class ToolsExecutor:  # matched by class name, as the real one is
        pass

    class LLM:
        pass

    ex, llm, bridge = ToolsExecutor(), LLM(), object()
    loop = _Loop([ex, llm])
    p = _Pipeline([object(), object(), llm, loop])
    insert_after_tool_executor(p, bridge)
    assert loop.elements == [ex, bridge, llm]
    with pytest.raises(PipelineShapeError):
        insert_after_tool_executor(_Pipeline([_Loop([llm])]), bridge)


# ------------------------------------------- integration with the real package


def test_real_pipeline_sees_selection_on_the_next_model_call(tmp_path):
    """With the real AgentDojo pipeline, the bridge sits between the tool
    executor and the next LLM turn: the second model call receives the rendered
    selection, and the baseline pipeline receives the raw tool output."""
    pytest.importorskip("agentdojo", reason="agentdojo extra not installed")
    from pydantic import BaseModel

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.types import text_content_block_from_string

    class Args(BaseModel):
        q: str

    def lookup(q: str) -> str:
        """Return a long tool output.

        :param q: the query
        """
        return "RAW TOOL OUTPUT for " + q

    class ScriptedLLM(BasePipelineElement):
        name = "gemini-2.5-flash-preview-04-17"

        def __init__(self):
            self.seen: list[list] = []

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            self.seen.append([dict(m) for m in messages])
            if len(self.seen) == 1:
                call = FunctionCall(function="lookup", args={"q": "x"}, id="c1")
                out = {"role": "assistant", "content": None, "tool_calls": [call]}
            else:
                out = {"role": "assistant", "content": [text_content_block_from_string("final")], "tool_calls": None}
            return query, runtime, env, [*messages, out], extra_args or {}

    def build(with_bridge: bool):
        llm = ScriptedLLM()
        pipeline = AgentPipeline.from_config(
            PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message="SYS")
        )
        transport = RecordingTransport(lambda req: selector_response(req, rendered="SELECTED"))
        if with_bridge:
            bridge = NeedlepathToolResultBridge(
                identity=IDENTITY, selector=SELECTOR, budget=BUDGET, transport=transport,
                telemetry=TelemetryWriter(tmp_path / f"needlepath_metrics_shard_{int(with_bridge)}.jsonl"),
                case_context=fixed_case_context(),
            )
            insert_after_tool_executor(pipeline, bridge)
        runtime = FunctionsRuntime([])
        runtime.register_function(lookup)
        pipeline.query("do the task", runtime)
        return llm, transport

    llm_b, transport_b = build(False)
    llm_r, transport_r = build(True)
    tool_b = [m for m in llm_b.seen[1] if m["role"] == "tool"][0]["content"][0]["content"]
    tool_r = [m for m in llm_r.seen[1] if m["role"] == "tool"][0]["content"][0]["content"]
    assert tool_b == "RAW TOOL OUTPUT for x" and transport_b.requests == []
    assert tool_r == "SELECTED" and transport_r.requests[0]["text"] == "RAW TOOL OUTPUT for x"
    assert "SYS" not in json.dumps(transport_r.requests)
