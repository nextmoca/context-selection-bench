#!/usr/bin/env python3
"""Multi-suite QA driver: SQuAD v2 / BFCL / TruthfulQA through the arm contract.

This is the port of the reference before/after context-reduction driver. It runs
short-answer QA cases through the uniform arm contract and scores them with each
suite's reference metric (the benchmark's published metric, reimplemented here).
The one structural change from the reference version:
the reference per-item engine call (which built a selection service and selected
context in-process) is replaced by ``arm.select(request) -> ContextResponse``.
Every downstream field the reference row carried is now read straight off
``ContextResponse``; the model prompt template and every suite's scoring formula
are byte-faithful so published numbers reproduce.

Per case ``(context, query)`` the driver:

  1. builds a per-suite ``ContextRequest`` (the case context as one
     ``external_data`` record for SQuAD/TruthfulQA, or one ``tool_result`` record
     per candidate function for BFCL);
  2. runs ``arm.select(request)`` to obtain the reduced (``after``) context plus
     the token/fallback/safety deltas;
  3. assembles the model prompt with ``qa_common.build_prompt(rendered_context,
     query)`` (SQuAD/BFCL) or the suite-specific MC1 prompt (TruthfulQA);
  4. calls the model via ``csbench.openai_client.call_openai`` (temperature 0);
  5. scores with the suite's ``evaluate_*`` from ``csbench.suites.{squad, bfcl,
     truthfulqa}``.

Arm mapping (the before/after axis): the ``full_context`` arm is the *before*
(the whole context, returned verbatim) and the hosted ``needlepath`` arm is the
*after* (the reduced selection). ``llmlingua2`` and ``cpc`` may also be named as
whole-context compression arms. For SQuAD v2 the before/after comparison is
intrinsic to a single arm call, the model answers the same question once over
the full passage and once over ``rendered_context``, so running the
``full_context`` arm yields a trivial-preservation control while ``needlepath``
yields the reduced-context answer scored against it.

The driver persists per-item rows (one JSON per case + a ``rows.json`` +
``summary.json`` per ``arm x suite``, carrying the same fields the reference
row carried, read off ``ContextResponse``), a run-level summary + markdown, a git
provenance snapshot, and a sha256 output manifest over every per-item file (via
``csbench.provenance``) so a run is independently verifiable.

OUTPUT SCHEMA
-------------
Per-item row (``qa_common.CaseResult`` as JSON), identical across all suites:
  ``case_id``, ``dataset``, ``mode``, ``original_tokens``, ``compressed_tokens``,
  ``compression_ratio``, ``fallback_used``, ``selection_safe``,
  ``fallback_reason``, ``selected_records``, ``available_records``,
  ``exact_match``, ``f1_score``, ``contains_ground_truth``, ``judge_score``,
  ``judge_reasoning``, ``accuracy_preserved``, ``latency_original_ms``,
  ``latency_compressed_ms``, ``response_original``, ``response_compressed``.
  The token/fallback/safety fields are read directly off ``ContextResponse``
  (``tokens_before`` / ``tokens_after`` / ``reduction_ratio`` /
  ``fallback_used`` / ``records_selected`` / ``records_available`` /
  ``safety.selection_safe`` / ``safety.fallback_reason``).

Per ``arm x suite`` summary (``qa_common.summarize``):
  ``cases``, ``passed``, ``failed``, ``accuracy_preservation_rate``,
  ``avg_compression_ratio``, ``total_original_tokens``,
  ``total_compressed_tokens``, ``total_tokens_saved``, ``fallbacks``,
  ``avg_f1``, ``avg_latency_compressed_ms``.

Run-level artifacts (at ``--out`` root):
  ``run_config.json`` (invocation + knobs + provenance), ``git_provenance.json``
  (neutral git snapshot), ``summary.json`` / ``summary.md`` (per ``arm x suite``
  rollup), and ``manifest.sha256`` (``<sha256>  <relpath>`` over every per-item
  output file, base-dir = ``--out``).

Layout on disk::

    <out>/<arm>/<suite_dir>/NNN_<case_id>.json   # one per case
    <out>/<arm>/<suite_dir>/rows.json            # all rows for the cell
    <out>/<arm>/<suite_dir>/summary.json         # the cell summary
    <out>/{run_config,summary,git_provenance}.json
    <out>/summary.md
    <out>/manifest.sha256

The ``datasets``- and ``openai``-backed pieces are imported lazily (inside the
functions that need them, and the SDK inside ``openai_client``), so importing
this module never requires those optional dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from csbench.arms import FullContextArm, NeedlepathArm
from csbench.arms.base import ContextArm
from csbench.contracts import (
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    TaskSpec,
)
from csbench.openai_client import build_openai_client, call_openai
from csbench.provenance import json_safe, probe_git_state, write_output_manifest
from csbench.suites.qa_common import (
    CaseResult,
    EvalCase,
    summary_markdown,
    write_suite_outputs,
)

# --------------------------------------------------------------------------- #
# Defaults (match the reference before/after run)
# --------------------------------------------------------------------------- #

SUPPORTED_SUITES = ("squad", "bfcl", "truthfulqa")
SUPPORTED_ARMS = ("full_context", "needlepath", "llmlingua2", "cpc")

# suite name -> on-disk directory (byte-faithful to the reference writer) and the
# human label used in the markdown rollup.
SUITE_DIRS = {"squad": "squad_v2", "bfcl": "bfcl", "truthfulqa": "truthfulqa"}
SUITE_TABLE_LABELS = {"squad": "SQuAD v2", "bfcl": "BFCL", "truthfulqa": "TruthfulQA MC1"}

DEFAULT_SUITES_ARG = ",".join(SUPPORTED_SUITES)
DEFAULT_ARMS_ARG = "full_context,needlepath"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_OPERATING_POINT = "np-2026-07-r1"
DEFAULT_SEED = 42

# Conservative default $/1k-token blended rate for the pre-run budget estimate.
# Deliberately high so the hard-stop over-estimates and never under-bills a run.
# Override via BENCH_USD_PER_1K_TOKENS. This is a safety guard, not billing.
DEFAULT_USD_PER_1K_TOKENS = 0.015

# Conservative per-case upper-bound *context* tokens per suite, for the pre-flight
# spend estimate only (evaluated before any dataset load or API call).
SUITE_CONTEXT_TOKENS = {"squad": 1200, "bfcl": 4096, "truthfulqa": 512}
# Context-bearing model calls per case: SQuAD answers before AND after; BFCL
# answers once and may call the judge once; TruthfulQA answers once.
SUITE_MODEL_CALLS = {"squad": 2, "bfcl": 2, "truthfulqa": 1}

# TruthfulQA operating point: the MC1 prompt is the load-bearing task text, not a
# compressible document, so the arm is a correct no-op: the budget is set well
# above any MC1 prompt so nothing is ever dropped.
TRUTHFULQA_MAX_CONTEXT_TOKENS = 8192

TRUTHFULQA_TOOL_NAME = "answer_truthfulqa_mc1"
TRUTHFULQA_SOURCE = "truthfulqa"


# --------------------------------------------------------------------------- #
# Arm construction (by name): mirrors the RULER harness's builder
# --------------------------------------------------------------------------- #


def build_arm(
    name: str,
    *,
    needlepath_url: str | None = None,
    operating_point: str | None = None,
    request_timeout_ms: int = 120000,
) -> ContextArm:
    """Construct an arm by name.

    ``full_context`` and ``needlepath`` are light and always available.
    ``llmlingua2`` and ``cpc`` carry heavy optional dependencies and are
    imported lazily so merely constructing the common arms never pulls in the
    model stacks. The hosted ``needlepath`` arm reads its bearer token from
    ``NEEDLEPATH_API_KEY`` in the environment only.
    """
    if name == "full_context":
        return FullContextArm()
    if name == "needlepath":
        if not needlepath_url:
            raise ValueError("the 'needlepath' arm requires --needlepath-url")
        return NeedlepathArm(
            base_url=needlepath_url,
            api_key=os.environ.get("NEEDLEPATH_API_KEY"),
            operating_point=operating_point,
            timeout_s=max(1.0, request_timeout_ms / 1000.0),
        )
    if name == "llmlingua2":
        from csbench.arms.llmlingua2 import Llmlingua2Arm  # optional heavy dep

        return Llmlingua2Arm()
    if name == "cpc":
        from csbench.arms.cpc import CpcArm  # optional heavy dep

        return CpcArm()
    raise ValueError(f"unknown arm: {name!r}")


# --------------------------------------------------------------------------- #
# TruthfulQA MC1: request construction + per-case evaluation
# --------------------------------------------------------------------------- #


def truthfulqa_budget(operating_point: str | None = None) -> BudgetSpec:
    """The TruthfulQA operating point.

    A fixed budget set well above any MC1 prompt so the arm never drops the
    options: the MC1 prompt is the load-bearing task text, not a compressible
    document, so context reduction is a correct no-op here.
    """
    return BudgetSpec(
        max_context_tokens=TRUTHFULQA_MAX_CONTEXT_TOKENS,
        operating_point=operating_point,
        mode="fixed",
        require_evidence_coverage=False,
    )


def build_truthfulqa_request(case: EvalCase, *, budget: BudgetSpec) -> ContextRequest:
    """Build the ``ContextRequest`` for one TruthfulQA MC1 case.

    The full MC1 prompt (question + lettered options + answer-format
    instruction) is a single ``external_data`` record; keywords are extracted
    from the question only, so no gold letter leaks into the selection signal.
    The arm runs for uniform token/fallback accounting, but the model prompt is
    always the suite-specific MC1 prompt (see ``evaluate_truthfulqa_row``), so
    the options ship intact regardless of arm.
    """
    from csbench.suites import squad  # lazy: pulls in `datasets` transitively

    query_terms = list(squad.extract_query_terms(case.query))
    record = ContextRecord(
        text=case.context,
        kind="external_data",
        id=f"{case.id}-mc1",
        source=TRUTHFULQA_SOURCE,
        title="TruthfulQA MC1 prompt",
        importance=5.0,
        keywords=query_terms,
        tags=["truthfulqa", "mc1", "multiple_choice"],
    )
    task = TaskSpec(
        prompt=f"Question: {case.query}",
        tool_name=TRUTHFULQA_TOOL_NAME,
        keywords=query_terms,
        tags=["truthfulqa", "mc1"],
    )
    return ContextRequest(
        request_id=f"truthfulqa:{case.id}",
        records=[record],
        task=task,
        budget=budget,
        render=True,
        render_format="plain",
        return_per_record=True,
    )


def evaluate_truthfulqa_row(
    arm: ContextArm,
    case: EvalCase,
    *,
    client: Any,
    model: str,
    max_tokens: int,
    budget: BudgetSpec,
) -> CaseResult:
    """Run one TruthfulQA MC1 case: select (no-op), answer, score.

    The arm reduces the request for uniform token/fallback accounting, then the
    model answers the *suite-specific* MC1 prompt (question + lettered options),
    which is byte-identical to ``case.context`` and independent of the arm's
    rendered output: this preserves MC1 scoring exactly (the deterministic
    single-correct check in ``truthfulqa.evaluate_truthfulqa_case``). The result
    is mapped into the shared ``CaseResult`` schema so every suite's rows agree.
    """
    from csbench.suites import truthfulqa  # lazy: pulls in `datasets` transitively

    request = build_truthfulqa_request(case, budget=budget)
    response: ContextResponse = arm.select(request)

    choices = list(case.metadata.get("choices") or [])
    prompt = truthfulqa.build_truthfulqa_prompt(case.query, choices)
    started = time.perf_counter()
    model_response = call_openai(client, model=model, prompt=prompt, max_tokens=max_tokens)
    latency = (time.perf_counter() - started) * 1000
    score = truthfulqa.evaluate_truthfulqa_case(model_response, case)

    safety = response.safety
    return CaseResult(
        case_id=case.id,
        dataset=case.dataset,
        mode="mc1",
        original_tokens=response.tokens_before,
        compressed_tokens=response.tokens_after,
        compression_ratio=response.reduction_ratio,
        fallback_used=bool(response.fallback_used),
        selection_safe=None if safety is None else safety.selection_safe,
        fallback_reason="" if safety is None else safety.fallback_reason,
        selected_records=int(response.records_selected),
        available_records=int(response.records_available),
        exact_match=score.correct,
        f1_score=score.score,
        contains_ground_truth=score.correct,
        judge_score=None,
        judge_reasoning=(
            f"parsed={score.parsed_letter or ''} "
            f"correct={score.correct_letter} malformed={score.malformed}"
        ),
        accuracy_preserved=score.correct,
        latency_original_ms=0.0,
        latency_compressed_ms=latency,
        response_original="",
        response_compressed=model_response,
    )


# --------------------------------------------------------------------------- #
# Case loading + per-suite runners (lazy suite imports)
# --------------------------------------------------------------------------- #


def load_cases(suite: str, n: int) -> list[EvalCase]:
    """Load the first ``n`` cases for ``suite`` from its pinned dataset source."""
    if suite == "squad":
        from csbench.suites import squad

        return squad.load_squad_cases(n)
    if suite == "bfcl":
        from csbench.suites import bfcl

        return bfcl.load_bfcl_cases(n)
    if suite == "truthfulqa":
        from csbench.suites import truthfulqa

        return truthfulqa.load_truthfulqa_cases(n)
    raise ValueError(f"unknown suite: {suite!r}")


def _log_case(label: str, index: int, total: int, result: CaseResult) -> None:
    print(
        f"[{label} {index}/{total}] "
        f"{'PASS' if result.accuracy_preserved else 'FAIL'} "
        f"reduction={result.compression_ratio:.1%} fallback={result.fallback_used}",
        file=sys.stderr,
    )


def run_squad_suite(
    arm: ContextArm,
    cases: Sequence[EvalCase],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    operating_point: str | None,
) -> list[CaseResult]:
    """Run every SQuAD case through ``squad.evaluate_squad_case`` (before/after).

    The model client is injected as a ``prompt -> answer`` callable so the suite
    module never imports a provider SDK; ``call_openai`` pins temperature 0.
    """
    from csbench.suites import squad

    def answer_fn(prompt: str) -> str:
        return call_openai(client, model=model, prompt=prompt, max_tokens=max_tokens)

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        result = squad.evaluate_squad_case(
            arm, case, answer_fn=answer_fn, operating_point=operating_point
        )
        results.append(result)
        _log_case("SQuAD", index, len(cases), result)
    return results


def run_bfcl_suite(
    arm: ContextArm,
    cases: Sequence[EvalCase],
    *,
    client: Any,
    judge_client: Any,
    model: str,
    judge_model: str,
    max_tokens: int,
    operating_point: str | None,
    use_judge: bool,
) -> list[CaseResult]:
    """Run every BFCL case through ``bfcl.evaluate_bfcl_case`` (ground-truth mode
    with an optional LLM judge)."""
    from csbench.suites import bfcl

    budget = bfcl.bfcl_budget(operating_point=operating_point)
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        result = bfcl.evaluate_bfcl_case(
            arm,
            client,
            judge_client,
            case,
            model=model,
            judge_model=judge_model,
            max_tokens=max_tokens,
            budget=budget,
            use_judge=use_judge,
        )
        results.append(result)
        _log_case("BFCL", index, len(cases), result)
    return results


def run_truthfulqa_suite(
    arm: ContextArm,
    cases: Sequence[EvalCase],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    operating_point: str | None,
) -> list[CaseResult]:
    """Run every TruthfulQA MC1 case through ``evaluate_truthfulqa_row``."""
    budget = truthfulqa_budget(operating_point)
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        result = evaluate_truthfulqa_row(
            arm, case, client=client, model=model, max_tokens=max_tokens, budget=budget
        )
        results.append(result)
        _log_case("TruthfulQA", index, len(cases), result)
    return results


def run_arm_suite(
    arm: ContextArm,
    suite: str,
    cases: Sequence[EvalCase],
    *,
    client: Any,
    judge_client: Any,
    model: str,
    judge_model: str,
    max_tokens: int,
    operating_point: str | None,
    use_judge: bool,
) -> list[CaseResult]:
    """Dispatch one ``(arm, suite)`` cell to its suite runner."""
    if suite == "squad":
        return run_squad_suite(
            arm, cases, client=client, model=model,
            max_tokens=max_tokens, operating_point=operating_point,
        )
    if suite == "bfcl":
        return run_bfcl_suite(
            arm, cases, client=client, judge_client=judge_client, model=model,
            judge_model=judge_model, max_tokens=max_tokens,
            operating_point=operating_point, use_judge=use_judge,
        )
    if suite == "truthfulqa":
        return run_truthfulqa_suite(
            arm, cases, client=client, model=model,
            max_tokens=max_tokens, operating_point=operating_point,
        )
    raise ValueError(f"unknown suite: {suite!r}")


# --------------------------------------------------------------------------- #
# Budget hard-stop (mirrors csbench.suites.ruler.harness.enforce_budget)
# --------------------------------------------------------------------------- #


def estimate_cost_usd(
    *,
    arms: Sequence[str],
    suites: Sequence[str],
    n: int,
    max_tokens: int,
    usd_per_1k: float,
) -> float:
    """Conservative upper-bound cost estimate for the whole run.

    Assumes every context-bearing model call sends a per-suite upper-bound
    context (``SUITE_CONTEXT_TOKENS``) plus ``max_tokens`` of output, across
    ``SUITE_MODEL_CALLS`` calls per case, for every arm and every case.
    Deliberately pessimistic so the hard-stop errs toward aborting rather than
    overspending; selection/compression arms in practice send fewer tokens.
    """
    total_tokens = 0
    for suite in suites:
        ctx = SUITE_CONTEXT_TOKENS.get(suite, max(SUITE_CONTEXT_TOKENS.values()))
        calls = SUITE_MODEL_CALLS.get(suite, 2)
        per_case = calls * (ctx + max_tokens)
        total_tokens += len(arms) * n * per_case
    return total_tokens / 1000.0 * usd_per_1k


def enforce_budget(
    *,
    arms: Sequence[str],
    suites: Sequence[str],
    n: int,
    max_tokens: int,
) -> float:
    """Estimate spend and abort (before any API call) if it would exceed the
    ``BENCH_BUDGET_USD`` env hard-stop. Returns the estimate. No env var set
    means no cap."""
    usd_per_1k = float(os.environ.get("BENCH_USD_PER_1K_TOKENS", DEFAULT_USD_PER_1K_TOKENS))
    estimate = estimate_cost_usd(
        arms=arms, suites=suites, n=n, max_tokens=max_tokens, usd_per_1k=usd_per_1k
    )
    cap_raw = os.environ.get("BENCH_BUDGET_USD")
    if cap_raw is not None and cap_raw.strip():
        cap = float(cap_raw)
        if estimate > cap:
            raise SystemExit(
                f"BENCH_BUDGET_USD hard-stop: estimated spend ${estimate:.2f} "
                f"exceeds cap ${cap:.2f} "
                f"(arms={len(arms)} x suites={len(suites)} x n={n}, "
                f"rate ${usd_per_1k:.4f}/1k tok). Lower --n / --suites / --arms or raise the cap."
            )
    return estimate


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace, *, estimated_spend_usd: float = 0.0) -> dict[str, Any]:
    """Run every ``(arm, suite)`` cell, persist all artifacts, return a manifest."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = build_openai_client()
    judge_client = build_openai_client()

    # All arms share the same examples per suite (loaded once).
    suite_cases: dict[str, list[EvalCase]] = {
        suite: load_cases(suite, args.n) for suite in args.suites
    }

    arm_objects = {
        name: build_arm(
            name,
            needlepath_url=args.needlepath_url,
            operating_point=args.operating_point,
            request_timeout_ms=args.request_timeout_ms,
        )
        for name in args.arms
    }

    config = {
        "run_name": args.run_name,
        "suites": list(args.suites),
        "arms": list(args.arms),
        "n": args.n,
        "model": args.model,
        "judge_model": args.judge_model,
        "use_judge": not args.no_judge,
        "max_tokens": args.max_tokens,
        "operating_point": args.operating_point or "",
        "needlepath_url": args.needlepath_url or "",
        "seed": args.seed,
        "estimated_spend_usd": estimated_spend_usd,
        "out": str(out),
        "command": " ".join(sys.argv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "run_config.json").write_text(
        json.dumps(json_safe(config), indent=2), encoding="utf-8"
    )
    (out / "git_provenance.json").write_text(
        json.dumps(json_safe(probe_git_state()), indent=2), encoding="utf-8"
    )

    output_files: list[Path] = []
    run_suites: dict[str, Any] = {}

    for arm_name in args.arms:
        arm = arm_objects[arm_name]
        for suite in args.suites:
            cases = suite_cases[suite]
            results = run_arm_suite(
                arm,
                suite,
                cases,
                client=client,
                judge_client=judge_client,
                model=args.model,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                operating_point=args.operating_point,
                use_judge=not args.no_judge,
            )
            suite_dir_name = SUITE_DIRS[suite]
            arm_out = out / arm_name
            summary = write_suite_outputs(arm_out, suite_dir_name, results)
            summary["arm"] = arm_name
            summary["suite"] = SUITE_TABLE_LABELS[suite]
            run_suites[f"{arm_name} / {SUITE_TABLE_LABELS[suite]}"] = summary
            output_files.extend(sorted((arm_out / suite_dir_name).glob("*.json")))

    run_summary = {
        "model": args.model,
        "judge_model": args.judge_model,
        "limit": args.n,
        "arms": list(args.arms),
        "operating_point": args.operating_point or "",
        "suites": run_suites,
    }
    (out / "summary.json").write_text(
        json.dumps(json_safe(run_summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "summary.md").write_text(summary_markdown(run_summary), encoding="utf-8")

    manifest_path = write_output_manifest(
        out / "manifest.sha256", output_files, base_dir=out
    )

    return {
        "out": str(out),
        "run_name": args.run_name,
        "arms": list(args.arms),
        "suites": list(args.suites),
        "n_cells": len(run_suites),
        "n_output_files": len(output_files),
        "summary": str(out / "summary.json"),
        "output_manifest": str(manifest_path),
        "estimated_spend_usd": estimated_spend_usd,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SQuAD v2 / BFCL / TruthfulQA cases through context-selection arms end-to-end."
    )
    parser.add_argument("--suites", default=DEFAULT_SUITES_ARG,
                        help="comma-separated suites: squad,bfcl,truthfulqa")
    parser.add_argument("--arms", default=DEFAULT_ARMS_ARG,
                        help="comma-separated arm names: full_context,needlepath,llmlingua2,cpc")
    parser.add_argument("--n", type=int, default=100, help="cases per suite")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-judge", action="store_true",
                        help="BFCL: use the deterministic argument-value check instead of the LLM judge")
    parser.add_argument("--needlepath-url", default=None,
                        help="base URL of the hosted Needlepath endpoint (required for the needlepath arm)")
    parser.add_argument("--operating-point", default=DEFAULT_OPERATING_POINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("runs/qa"))
    parser.add_argument("--request-timeout-ms", type=int, default=120000)
    parser.add_argument("--run-name",
                        default=f"qa_{datetime.now().strftime('%Y%m%d')}")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.suites = _parse_csv(args.suites)
    args.arms = _parse_csv(args.arms)

    unknown_suites = [s for s in args.suites if s not in SUPPORTED_SUITES]
    if unknown_suites:
        raise SystemExit(f"Unknown suites: {', '.join(sorted(unknown_suites))}")
    if not args.suites:
        raise SystemExit("--suites must name at least one suite")
    unknown_arms = [a for a in args.arms if a not in SUPPORTED_ARMS]
    if unknown_arms:
        raise SystemExit(f"Unknown arms: {', '.join(sorted(unknown_arms))}")
    if not args.arms:
        raise SystemExit("--arms must name at least one arm")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if "needlepath" in args.arms and not args.needlepath_url:
        raise SystemExit("the 'needlepath' arm requires --needlepath-url")

    # Estimate spend and abort BEFORE constructing any client or making any call.
    estimate = enforce_budget(
        arms=args.arms, suites=args.suites, n=args.n, max_tokens=args.max_tokens
    )
    print(f"estimated spend upper bound: ${estimate:.2f}", file=sys.stderr)

    result = run(args, estimated_spend_usd=estimate)
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
