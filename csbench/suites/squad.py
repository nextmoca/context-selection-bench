"""SQuAD v2 before/after context-reduction surface.

This module provides the SQuAD v2 slice of the QA benchmark: loading official
validation examples, building the per-case selection request, and scoring one
case as a *before/after* comparison: the model answers the same question once
over the full context and once over the arm's reduced context, and the two
answers are compared with the suite's F1/exact-match scorers.

The single per-item engine call in the reference implementation is replaced by
the uniform arm contract ``arm.select(request) -> ContextResponse``; the model
client is injected as a callable so this module never imports a provider SDK.
Shared QA primitives (``EvalCase``/``CaseResult`` and the F1/EM scorers) live in
``csbench.suites.qa_common`` so the SQuAD and BFCL surfaces score identically.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from csbench.arms.base import ContextArm
from csbench.contracts import (
    AdaptiveBudget,
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    TaskSpec,
)
from csbench.suites.qa_common import (
    CaseResult,
    EvalCase,
    compute_exact_match,
    compute_f1,
)

# --------------------------------------------------------------------------- #
# Dataset / operating-point constants (pin the fetch so numbers reproduce)
# --------------------------------------------------------------------------- #

SQUAD_DATASET_ID = "rajpurkar/squad_v2"
SQUAD_SPLIT = "validation"
# Leave ``None`` to track the dataset's default snapshot exactly as the
# reference run fetched it; set a commit hash here to lock a specific revision.
SQUAD_REVISION: str | None = None

SQUAD_SOURCE = "squad_v2"
SQUAD_TOOL_NAME = "answer_squad_question"

# Selection/budget operating point (mirrors the reference before/after run).
DEFAULT_MAX_SELECTED_TOKENS = 4096
DEFAULT_MAX_SELECTED_RECORDS = 40
DEFAULT_MAX_EXCERPT_TOKENS = 512

# Before/after preservation thresholds (byte-faithful to the reference scorer).
_F1_PRESERVED_THRESHOLD = 0.7
_CONTAINS_GT_F1_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Prompt template (byte-faithful, governs the reproduced answers)
# --------------------------------------------------------------------------- #


def build_squad_prompt(context: str, query: str) -> str:
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""


# --------------------------------------------------------------------------- #
# Selection keyword extraction
# --------------------------------------------------------------------------- #


def extract_query_terms(text: str) -> tuple[str, ...]:
    """Deterministic query-term extraction used to seed selection keywords.

    Lowercases, drops a small generic QA stopword set, dedups preserving order,
    and caps at 32 terms, byte-faithful to the reference implementation.
    """
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
    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_/-]{2,}", text.lower()):
        if raw in stop or raw in seen:
            continue
        seen.add(raw)
        terms.append(raw)
        if len(terms) >= 32:
            break
    return tuple(terms)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_squad_cases(limit: int, *, revision: str | None = SQUAD_REVISION) -> list[EvalCase]:
    """Load the first ``limit`` answerable SQuAD v2 validation examples.

    Skips examples with no gold answer (SQuAD v2's unanswerable rows) and keeps
    the first gold answer as the reference, faithful to the reference loader.
    """
    from datasets import load_dataset  # optional "qa" extra

    rows = load_dataset(SQUAD_DATASET_ID, split=SQUAD_SPLIT, revision=revision)
    cases: list[EvalCase] = []
    for item in rows:
        answers = item.get("answers") or {}
        answer_texts = answers.get("text") or []
        if not answer_texts:
            continue
        cases.append(
            EvalCase(
                id=str(item.get("id") or f"squad_{len(cases)}"),
                dataset="squad_v2",
                context=str(item.get("context") or ""),
                query=str(item.get("question") or ""),
                ground_truth=str(answer_texts[0]),
                metadata={"title": item.get("title"), "source": "SQuAD v2"},
            )
        )
        if len(cases) >= limit:
            break
    return cases


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def default_squad_budget(operating_point: str | None = None) -> BudgetSpec:
    """The shared SQuAD operating point (adaptive with the reference ladder)."""
    return BudgetSpec(
        max_context_tokens=DEFAULT_MAX_SELECTED_TOKENS,
        operating_point=operating_point,
        max_records=DEFAULT_MAX_SELECTED_RECORDS,
        max_excerpt_tokens_per_record=DEFAULT_MAX_EXCERPT_TOKENS,
        mode="adaptive",
        adaptive=AdaptiveBudget(
            initial_tokens=DEFAULT_MAX_SELECTED_TOKENS,
            escalation_tokens=[DEFAULT_MAX_SELECTED_TOKENS * 2],
            allow_full_context_fallback=True,
        ),
        require_evidence_coverage=True,
    )


def build_squad_request(case: EvalCase, *, budget: BudgetSpec) -> ContextRequest:
    """Build the ``ContextRequest`` for one SQuAD case.

    The whole passage is a single ``external_data`` record; the arm sub-selects
    (or, on fallback, returns it whole). Keywords are extracted from the
    question only, so no gold answer leaks into the selection signal.
    """
    query_terms = list(extract_query_terms(case.query))
    record = ContextRecord(
        text=case.context,
        kind="external_data",
        id=f"{case.id}-context",
        source=SQUAD_SOURCE,
        title="SQuAD v2 context",
        importance=5.0,
        keywords=query_terms,
        tags=["squad_v2", "qa_context", "factoid_qa"],
    )
    task = TaskSpec(
        prompt=f"Question: {case.query}",
        tool_name=SQUAD_TOOL_NAME,
        keywords=query_terms,
        tags=["squad_v2", "qa", "factoid_qa"],
    )
    return ContextRequest(
        request_id=f"squad_v2:{case.id}",
        records=[record],
        task=task,
        budget=budget,
        render=True,
        render_format="plain",
        return_per_record=True,
    )


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #


def evaluate_squad_case(
    arm: ContextArm,
    case: EvalCase,
    *,
    answer_fn: Callable[[str], str],
    budget: BudgetSpec | None = None,
    operating_point: str | None = None,
) -> CaseResult:
    """Score one SQuAD case as a before/after comparison.

    Steps:

      1. ``arm.select(request)`` reduces the passage, yielding a rendered context
         and the token/fallback/safety fields on ``ContextResponse``;
      2. the model (``answer_fn``, a ``prompt -> answer`` callable, so no
         provider SDK is imported here) answers the same question once over the
         full passage and once over the reduced context;
      3. the two answers are compared with the suite's exact-match / F1 scorers,
         and preservation is declared when the answers agree closely or the
         reduced-context answer still contains the gold answer.

    Token deltas, fallback flag, selected/available record counts, and the
    neutral safety subset are read straight off ``ContextResponse``.
    """
    request = build_squad_request(
        case, budget=budget or default_squad_budget(operating_point)
    )
    response: ContextResponse = arm.select(request)
    compressed_context = response.rendered_context

    started = time.perf_counter()
    response_original = answer_fn(build_squad_prompt(case.context, case.query))
    latency_original = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    response_compressed = answer_fn(build_squad_prompt(compressed_context, case.query))
    latency_compressed = (time.perf_counter() - started) * 1000

    exact = compute_exact_match(response_original, response_compressed)
    f1 = compute_f1(response_original, response_compressed)
    contains_gt = (
        case.ground_truth.lower() in response_compressed.lower()
        or compute_f1(response_compressed, case.ground_truth) > _CONTAINS_GT_F1_THRESHOLD
    )
    preserved = f1 > _F1_PRESERVED_THRESHOLD or contains_gt is True

    safety = response.safety
    return CaseResult(
        case_id=case.id,
        dataset=case.dataset,
        mode="before-after",
        original_tokens=response.tokens_before,
        compressed_tokens=response.tokens_after,
        compression_ratio=response.reduction_ratio,
        fallback_used=bool(response.fallback_used),
        selection_safe=None if safety is None else safety.selection_safe,
        fallback_reason="" if safety is None else safety.fallback_reason,
        selected_records=int(response.records_selected),
        available_records=int(response.records_available),
        exact_match=exact,
        f1_score=f1,
        contains_ground_truth=contains_gt,
        judge_score=None,
        judge_reasoning="",
        accuracy_preserved=preserved,
        latency_original_ms=latency_original,
        latency_compressed_ms=latency_compressed,
        response_original=response_original,
        response_compressed=response_compressed,
    )
