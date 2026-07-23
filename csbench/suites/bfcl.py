"""BFCL (Berkeley Function-Calling) suite: the plain function-call-matching path.

This is the tool-calling half of the before/after context-reduction comparison.
For each BFCL "simple" item the driver reduces the candidate function schemas
with the arm under test, asks the model to pick the function and arguments from
the reduced context, and scores the answer against the public BFCL ground-truth
argument values (with an optional LLM-as-judge on top).

Only the plain function-call-matching surface lives here: case loading, the
question extractor, the deterministic ground-truth argument-recall check, and
the per-case evaluator. The single structural change from the reference version
is that the internal per-item engine call is replaced by the uniform arm
contract ``arm.select(request) -> ContextResponse``; everything downstream reads
the reduced context and the token/fallback deltas straight off the response. The
scoring formulas, prompt template, and ground-truth matching are byte-faithful
so the published numbers reproduce.

Shared records/scorers/writers come from :mod:`csbench.suites.qa_common`; the
answer/judge model calls come from :mod:`csbench.openai_client`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from csbench.contracts import (
    AdaptiveBudget,
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    TaskSpec,
)
from csbench.openai_client import call_openai, judge_openai
from csbench.suites.qa_common import (
    CaseResult,
    EvalCase,
    _download_json,
    build_prompt,
    estimate_context_tokens,
)

# --------------------------------------------------------------------------- #
# Dataset schema (pinned resolve URLs)
# --------------------------------------------------------------------------- #

# BFCL is served from the Gorilla "Berkeley-Function-Calling-Leaderboard" repo
# on the Hugging Face hub. The revision is factored into a single constant so the
# two ``resolve/<revision>/`` URLs below always agree and the exact snapshot a
# run fetched is recorded in one place; freeze it to a commit SHA to pin a run.
BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_DATASET_REVISION = "main"
_BFCL_RESOLVE = (
    f"https://huggingface.co/datasets/{BFCL_DATASET}/resolve/{BFCL_DATASET_REVISION}"
)
BFCL_DATA_URL = f"{_BFCL_RESOLVE}/BFCL_v3_simple.json"
BFCL_GT_URL = f"{_BFCL_RESOLVE}/possible_answer/BFCL_v3_simple.json"

# Selection/budget operating point for the BFCL cases (mirrors the reference
# adaptive adapter). BFCL does not require evidence coverage: the "answer" is a
# function/argument choice, not a span that must be quoted back.
DEFAULT_MAX_SELECTED_TOKENS = 4096
DEFAULT_MAX_SELECTED_RECORDS = 40
DEFAULT_MAX_EXCERPT_TOKENS = 512

BFCL_TOOL_NAME = "bfcl"
BFCL_SOURCE = "bfcl"


# --------------------------------------------------------------------------- #
# Selection-keyword helper (engine-free, pure text)
# --------------------------------------------------------------------------- #


def _simple_terms(text: str) -> tuple[str, ...]:
    import re

    stop = {
        "about",
        "after",
        "answer",
        "based",
        "before",
        "following",
        "function",
        "question",
        "should",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    terms = []
    seen = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_/-]{2,}", text.lower()):
        if raw in stop or raw in seen:
            continue
        seen.add(raw)
        terms.append(raw)
        if len(terms) >= 32:
            break
    return tuple(terms)


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #


def _bfcl_question(item: dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, list) and question:
        first = question[0]
        if isinstance(first, list) and first:
            message = first[0]
            if isinstance(message, dict):
                return str(message.get("content") or "")
        if isinstance(first, dict):
            return str(first.get("content") or "")
    return str(question or "")


def load_bfcl_cases(limit: int) -> list[EvalCase]:
    """Load the first ``limit`` BFCL "simple" items and their ground-truth
    argument sets from the pinned ``resolve/<revision>/`` URLs.

    Each case's ``context`` carries the candidate function schemas as pretty,
    key-sorted JSON (the reducible tool context); ``query`` carries the user
    request; ``ground_truth`` carries the public expected function-call values as
    a JSON string (``[]`` when none is published for the id).
    """
    rows = _download_json(BFCL_DATA_URL)
    answers = _download_json(BFCL_GT_URL)
    answer_by_id = {
        str(item.get("id")): item.get("ground_truth")
        for item in answers
        if isinstance(item, dict)
    }
    cases: list[EvalCase] = []
    for item in rows[:limit]:
        item_id = str(item.get("id") or f"bfcl_{len(cases)}")
        functions = item.get("function") or []
        cases.append(
            EvalCase(
                id=item_id,
                dataset="bfcl",
                context=json.dumps(functions, indent=2, sort_keys=True),
                query=_bfcl_question(item),
                ground_truth=json.dumps(answer_by_id.get(item_id) or []),
                metadata={
                    "source": "BFCL",
                    "category": "simple",
                    "dataset": BFCL_DATASET,
                    "dataset_revision": BFCL_DATASET_REVISION,
                },
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# Ground-truth argument-recall check (deterministic, byte-faithful)
# --------------------------------------------------------------------------- #


def check_bfcl_ground_truth(response: str, ground_truth: str) -> bool:
    try:
        gt_list = json.loads(ground_truth)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    response_lower = (response or "").lower()
    for gt_call in gt_list:
        if not isinstance(gt_call, dict):
            continue
        for _function_name, params in gt_call.items():
            if not isinstance(params, dict):
                continue
            values_found = 0
            values_total = 0
            for _param_name, accepted_values in params.items():
                if not isinstance(accepted_values, list):
                    accepted_values = [accepted_values]
                real_values = [value for value in accepted_values if value not in ("", None)]
                if not real_values:
                    continue
                values_total += 1
                for value in real_values:
                    value_text = str(value).lower()
                    if value_text in response_lower:
                        values_found += 1
                        break
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    variants = [
                        str(number),
                        f"{number:.0f}",
                        f"{number:g}",
                        str(int(number)) if number == int(number) else "",
                    ]
                    if any(variant and variant.lower() in response_lower for variant in variants):
                        values_found += 1
                        break
            if values_total == 0:
                return True
            if values_found / values_total >= 0.5:
                return True
    return False


# --------------------------------------------------------------------------- #
# Request construction (per BFCL case)
# --------------------------------------------------------------------------- #


def bfcl_budget(
    *,
    max_selected_tokens: int = DEFAULT_MAX_SELECTED_TOKENS,
    max_selected_records: int = DEFAULT_MAX_SELECTED_RECORDS,
    max_excerpt_tokens: int = DEFAULT_MAX_EXCERPT_TOKENS,
    operating_point: str | None = None,
) -> BudgetSpec:
    """The BFCL operating point: adaptive selection with a single escalation
    step and full-context fallback, and no evidence-coverage requirement (a
    function/argument choice, not a quoted span)."""
    return BudgetSpec(
        max_context_tokens=max_selected_tokens,
        operating_point=operating_point,
        max_records=max_selected_records,
        max_excerpt_tokens_per_record=max_excerpt_tokens,
        mode="adaptive",
        adaptive=AdaptiveBudget(
            initial_tokens=max_selected_tokens,
            escalation_tokens=[max_selected_tokens * 2],
            allow_full_context_fallback=True,
        ),
        require_evidence_coverage=False,
    )


def build_bfcl_request(case: EvalCase, *, budget: BudgetSpec) -> ContextRequest:
    """Build the ``ContextRequest`` for one BFCL case.

    The candidate function schemas are split into one record per function (the
    reference recorded each function as a separate tool-result state), each
    titled/sourced and keyworded from the query plus its own terms, and the
    task prompt asks for the correct function and arguments. ``hybrid`` render
    format mirrors the reference tool context format.
    """
    query_terms = _simple_terms(case.query)
    try:
        functions = json.loads(case.context)
    except json.JSONDecodeError:
        functions = [case.context]
    if not isinstance(functions, list):
        functions = [functions]

    records = []
    for index, function in enumerate(functions):
        function_text = json.dumps(function, sort_keys=True, indent=2)
        records.append(
            ContextRecord(
                text=function_text,
                kind="tool_result",
                id=f"{case.id}-function-{index}",
                source=BFCL_SOURCE,
                title=f"BFCL function schema {index + 1}",
                importance=3.0,
                keywords=[*query_terms, *_simple_terms(function_text)[:16]],
                tags=["bfcl", "function_schema", "tool_use"],
            )
        )

    task = TaskSpec(
        prompt=f"Select the correct function and arguments for this user request: {case.query}",
        tool_name=BFCL_TOOL_NAME,
        keywords=list(query_terms),
        tags=["bfcl", "tool_use"],
    )
    return ContextRequest(
        request_id=f"bfcl:{case.id}",
        records=records,
        task=task,
        budget=budget,
        render=True,
        render_format="hybrid",
        return_per_record=True,
    )


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #


def evaluate_bfcl_case(
    arm: Any,
    client: Any,
    judge_client: Any,
    case: EvalCase,
    *,
    model: str,
    judge_model: str,
    max_tokens: int,
    budget: BudgetSpec,
    use_judge: bool,
) -> CaseResult:
    """Run one BFCL case end-to-end: reduce with ``arm``, answer, score.

    The arm reduces the candidate function context (``arm.select(request)``);
    the model answers over the reduced context; and the answer is scored by the
    LLM judge (``>= 3`` of 5 counts as preserved) when ``use_judge`` is set,
    falling back to the deterministic argument-recall check on any judge error,
    or the deterministic check directly when ``use_judge`` is false.
    """
    request = build_bfcl_request(case, budget=budget)
    response: ContextResponse = arm.select(request)
    compressed_context = response.rendered_context

    original_tokens = estimate_context_tokens(case.context)
    compressed_tokens = estimate_context_tokens(compressed_context)
    started = time.perf_counter()
    model_response = call_openai(
        client,
        model=model,
        prompt=build_prompt(compressed_context, case.query),
        max_tokens=max_tokens,
    )
    latency = (time.perf_counter() - started) * 1000

    judge_score: float | None = None
    judge_reasoning = ""
    contains_gt: bool
    if use_judge:
        try:
            judge_score, judge_reasoning = judge_openai(
                judge_client,
                model=judge_model,
                question=case.query,
                ground_truth=case.ground_truth,
                prediction=model_response,
            )
            contains_gt = judge_score >= 3.0
        except Exception as exc:
            judge_reasoning = f"judge_error: {type(exc).__name__}: {exc}"
            contains_gt = check_bfcl_ground_truth(model_response, case.ground_truth)
    else:
        contains_gt = check_bfcl_ground_truth(model_response, case.ground_truth)

    safety = response.safety
    return CaseResult(
        case_id=case.id,
        dataset=case.dataset,
        mode="ground_truth",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=1 - (compressed_tokens / original_tokens) if original_tokens else 0.0,
        fallback_used=bool(response.fallback_used),
        selection_safe=None if safety is None else safety.selection_safe,
        fallback_reason="" if safety is None else safety.fallback_reason,
        selected_records=int(response.records_selected),
        available_records=int(response.records_available),
        exact_match=False,
        f1_score=(judge_score / 5.0) if judge_score is not None else 0.0,
        contains_ground_truth=contains_gt,
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
        accuracy_preserved=contains_gt is True,
        latency_original_ms=0.0,
        latency_compressed_ms=latency,
        response_original="",
        response_compressed=model_response,
    )
