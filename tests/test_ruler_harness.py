import csbench.suites.ruler.harness as H
from csbench.arms import FullContextArm, NeedlepathArm
from csbench.stub import StubServer
from csbench.suites.ruler.data import OfficialRulerExample
from csbench.suites.ruler.prompts import baseline_prompt, compression_prompt, needlepath_prompt


def _example() -> OfficialRulerExample:
    return OfficialRulerExample(
        task="niah_single_1",
        index=3,
        input_text="Haystack line one.\nThe special magic number is 8675309.\nQuestion: what is the number?",
        answer_prefix=" Answer:",
        expected_answer=("8675309",),
        length=8192,
    )


def _fake_model(captured):
    def fake_call(client, *, model, prompt, max_output_tokens):
        captured["prompt"] = prompt
        return {"content": "8675309", "input_tokens": 10, "output_tokens": 1, "latency_ms": 1.0}
    return fake_call


def test_prompt_templates_are_byte_exact():
    assert baseline_prompt("CTX", " A:") == "CTX A:"
    assert compression_prompt("CC", "Q?", " A:") == "CC\n\nQ?\n A:"
    assert needlepath_prompt("SEL", "Q?", " A:", fallback_used=False, input_text="RAW") == (
        "Use the selected official RULER context fragments below to answer the question. "
        "Return only the requested continuation.\n\n"
        "SELECTED OFFICIAL RULER CONTEXT:\nSEL\n\nQ?\n A:"
    )
    # fallback degrades to the bare baseline (input_text + answer_prefix)
    assert needlepath_prompt("SEL", "Q?", " A:", fallback_used=True, input_text="RAW") == "RAW A:"


def test_full_context_prompt_is_byte_identical_to_baseline(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(H, "call_gemini", _fake_model(captured))
    ex = _example()
    row = H.run_item(
        FullContextArm(), "full_context", ex,
        position=0, client=None, model="m", max_output_tokens=32,
        budget=H._budget_spec(None), operating_point=None,
    )
    # The control arm must reproduce the reference baseline byte-for-byte.
    assert captured["prompt"] == ex.input_text + ex.answer_prefix
    assert row["correct"] is True  # "8675309" matches expected


def test_needlepath_records_carry_no_keywords():
    # The reference recorded chunks with no explicit keywords (the engine infers
    # them from record text); selection keywords live only on the task. Putting
    # them on records changes the hosted selection, so guard against regressions.
    ex = _example()
    from csbench.suites.ruler.data import split_official_prompt

    context, question = split_official_prompt(ex.input_text)
    req = H.build_request("needlepath", ex, context=context, question=question, budget=H._budget_spec(None))
    assert all(not r.keywords for r in req.records)
    assert req.task.keywords  # the task DOES carry selection keywords


def test_needlepath_uses_scaffolded_template(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(H, "call_gemini", _fake_model(captured))
    ex = _example()
    with StubServer() as stub:
        arm = NeedlepathArm(base_url=stub.base_url, operating_point="np-2026-07-r1")
        H.run_item(
            arm, "needlepath", ex,
            position=0, client=None, model="m", max_output_tokens=32,
            budget=H._budget_spec("np-2026-07-r1"), operating_point="np-2026-07-r1",
        )
    # Stub passes context through (fallback_used=False) -> scaffolded template.
    assert captured["prompt"].startswith("Use the selected official RULER context fragments")
    assert "SELECTED OFFICIAL RULER CONTEXT:" in captured["prompt"]
