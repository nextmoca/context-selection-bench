import json

from csbench.contracts import (
    AdaptiveBudget,
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    GateSummary,
    SafetySummary,
    SelectedRecord,
    TaskSpec,
)


def _sample_request() -> ContextRequest:
    return ContextRequest(
        request_id="req-1",
        records=[
            ContextRecord(text="alpha", kind="external_data", id="r0", importance=1.0, keywords=["a"]),
            ContextRecord(text="beta", title="B", attributes={"chunk_index": 2}),
        ],
        task=TaskSpec(prompt="q?", tool_name="lookup", required_record_ids=["r0"], recent_prompts=["p0"]),
        budget=BudgetSpec(
            max_context_tokens=4096,
            operating_point="np-2026-07-r1",
            mode="adaptive",
            adaptive=AdaptiveBudget(initial_tokens=2048, escalation_tokens=[4096, 8192]),
            require_evidence_coverage=True,
        ),
    )


def test_request_roundtrip_through_json():
    req = _sample_request()
    wire = json.loads(json.dumps(req.to_wire()))
    back = ContextRequest.from_wire(wire)
    assert back == req
    # nested adaptive survives
    assert back.budget.adaptive is not None
    assert back.budget.adaptive.escalation_tokens == [4096, 8192]


def test_response_roundtrip_through_json():
    resp = ContextResponse(
        request_id="req-1",
        rendered_context="alpha\n\nbeta",
        policy_version="np-2026-07-r1",
        selected=[SelectedRecord(record_id="r0", score=12.0, reason="required", excerpt="alpha", selected_tokens=1)],
        tokens_before=10,
        tokens_after=6,
        tokens_saved=4,
        records_available=2,
        records_selected=1,
        fallback_used=False,
        engine_latency_ms=3.5,
        budget_tokens=4096,
        attempted_budget_tokens=[2048, 4096],
        reduction_ratio=0.4,
        safety=SafetySummary(selection_safe=True, evidence_shape="factoid_qa"),
        gate=GateSummary(engaged=True, reason="engage:needle", signals={"n_candidates": 5}),
    )
    wire = json.loads(json.dumps(resp.to_wire()))
    back = ContextResponse.from_wire(wire)
    assert back == resp
    assert back.gate is not None and back.gate.signals["n_candidates"] == 5
    assert back.safety is not None and back.safety.evidence_shape == "factoid_qa"


def test_response_null_safety_and_gate_roundtrip():
    resp = ContextResponse(request_id="x", rendered_context="c")
    back = ContextResponse.from_wire(json.loads(json.dumps(resp.to_wire())))
    assert back.safety is None
    assert back.gate is None
