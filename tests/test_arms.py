from csbench.arms import FullContextArm, NeedlepathArm
from csbench.arms.needlepath import ContractError
from csbench.contracts import BudgetSpec, ContextRecord, ContextRequest, TaskSpec
from csbench.stub import STUB_POLICY_VERSION, StubServer


def _request() -> ContextRequest:
    return ContextRequest(
        request_id="req-42",
        records=[
            ContextRecord(text="the capital is Lima", title="doc1", id="r0"),
            ContextRecord(text="unrelated filler", title="doc2", id="r1"),
        ],
        task=TaskSpec(prompt="what is the capital?"),
        budget=BudgetSpec(max_context_tokens=4096),
    )


def test_full_context_arm_passthrough():
    resp = FullContextArm().select(_request())
    assert resp.request_id == "req-42"
    assert "Lima" in resp.rendered_context and "filler" in resp.rendered_context
    assert resp.tokens_saved == 0
    assert resp.records_available == 2
    assert resp.records_selected == 2
    assert resp.fallback_used is False
    assert len(resp.selected) == 2


def test_needlepath_arm_against_stub():
    req = _request()
    with StubServer() as stub:
        arm = NeedlepathArm(base_url=stub.base_url, operating_point="np-2026-07-r1")
        resp = arm.select(req)
    assert resp.request_id == req.request_id
    assert resp.policy_version == STUB_POLICY_VERSION
    assert resp.records_available == 2
    assert "Lima" in resp.rendered_context
    assert "client_latency_ms" in resp.format_metrics


def test_needlepath_arm_rejects_mismatched_request_id():
    # Point the client at the stub but tamper the echoed id by monkeypatching.
    import csbench.stub.server as srv

    original = srv.build_stub_response

    def _wrong(request):
        r = original(request)
        r.request_id = "SOMETHING-ELSE"
        return r

    srv.build_stub_response = _wrong
    try:
        with StubServer() as stub:
            arm = NeedlepathArm(base_url=stub.base_url)
            try:
                arm.select(_request())
                raised = False
            except ContractError:
                raised = True
        assert raised, "client must reject a response whose request_id does not echo"
    finally:
        srv.build_stub_response = original
