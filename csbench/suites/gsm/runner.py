#!/usr/bin/env python3
"""GSM boundary-suite sweep runner: a DIAGNOSTIC, NOT a flagship result.

BOUNDARY-SUITE FRAMING (read this first). This suite exists to demonstrate two
disclosed BOUNDARY cases for context selection, NOT to headline a win:

  1. GSM8K -- clean, single-document math word problems. There is nothing to
     select away, so record-level selection is a NO-OP BY DESIGN: the arm
     correctly preserves the whole prompt.
  2. GSM-IC -- in-context distractors injected WITHIN a single record (one
     sentence woven into the narrative). These are OUTSIDE the selection regime,
     because record-level selection cannot remove a distractor sentence that
     lives inside the one record it must keep.

Flagship claims live with the RULER and the tool/document suites, not here. No
headline/marketing numbers belong in this module or the numbers it prints.

---------------------------------------------------------------------------
What this driver does
---------------------------------------------------------------------------

It runs a matched-token-budget sweep for either mode over one shared protocol:
sweep every arm to the SAME reduction targets (25% / 50% / 75%, plus each arm's
native default) and compare at equal budget. It is the port of the reference
matched-budget sweep plus the reference GSM8K / GSM-IC pilot entry points. The
one structural change from the reference version: the reference per-item engine
call (a method adapter that both prepared the selector input AND drove the
engine/compressor to produce a ready-to-send ``prompt_context``) is split into
(a) request construction via ``build_gsm_request`` and (b) the uniform arm
contract ``arm.select(request) -> ContextResponse``. The model prompt is then
reassembled from the arm's ``rendered_context`` exactly as the reference adapter
assembled it.

For each ``(arm, budget, condition, item)`` the driver:

  1. resolves the per-item token budget (``resolve_budget_tokens`` -- a % of the
     QUESTION content's token count, never the full prompt's, so the fixed
     few-shot scaffolding is excluded from the basis and the requested %
     constrains exactly the selectable content);
  2. builds a per-arm ``ContextRequest`` (``build_gsm_request``);
  3. runs ``arm.select(request)`` to obtain the reduced ``rendered_context``;
  4. reassembles the model prompt (``assemble_model_prompt``): the
     ``full_context`` arm's rendered context is byte-identical to the raw prompt
     and is sent verbatim; every other arm's rendered context is the reduced
     QUESTION content, so the identical fixed few-shot block is re-attached
     verbatim -- the same three-way split the reference used, keeping every
     method on identical few-shot scaffolding so none gets a setup edge;
  5. calls the model via ``csbench.openai_client.call_openai`` (temperature 0);
  6. scores GSM8K exact-match via ``csbench.stats.score_exact_match`` against the
     already-known ``gold_numeric``.

It aggregates per ``(arm x budget x condition)`` cell -- exact-match rate, token
reduction, fallback rate, latency, and (GSM-IC only) a per-BASE-PROBLEM
``macro_accuracy`` -- writes per-item rows, per-cell summaries, a combined
rollup, and a sha256 output manifest so a run is independently verifiable.

A ``BENCH_BUDGET_USD`` pre-flight hard stop aborts BEFORE any API call if the
estimated spend (from ``csbench.suites.gsm.cost``) would exceed the cap, mirroring
the other suites' spend guard. The ``datasets`` and ``openai`` packages are
OPTIONAL dependencies, imported lazily inside the loaders / model client, so this
module imports without them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from csbench.arms import FullContextArm, NeedlepathArm
from csbench.arms.base import ContextArm
from csbench.contracts import ContextResponse
from csbench.openai_client import build_openai_client, call_openai
from csbench.provenance import json_safe, probe_git_state, write_output_manifest
from csbench.stats import macro_accuracy, score_exact_match
from csbench.suites.gsm import cost
from csbench.suites.gsm.gsm8k_data import (
    DEFAULT_SEED,
    _assemble_prompt,
    build_fewshot_prompt,
    build_prompt_examples,
    load_gsm8k_test_items,
)
from csbench.suites.gsm.gsmic_data import (
    load_gsmic_2step_items,
    sample_gsmic_2step,
)
from csbench.suites.gsm.request import build_gsm_request, build_gsmic_adapter_items

# --------------------------------------------------------------------------- #
# Defaults / modes (match the reference sweeps)
# --------------------------------------------------------------------------- #

MODE_GSM8K = "gsm8k"
MODE_GSM_IC = "gsm_ic"
SUPPORTED_MODES = (MODE_GSM8K, MODE_GSM_IC)

SUPPORTED_ARMS = ("full_context", "needlepath", "llmlingua2", "cpc")
DEFAULT_ARMS = ("full_context", "needlepath", "llmlingua2")
BASELINE_ARM = "full_context"
NEEDLEPATH_ARM = "needlepath"

# Budget-label sets per mode (re-exported from the cost module so the sweep, the
# cost estimate, and the CLI agree on one definition). The GSM-IC sweep's
# established scope is 25% / 75% / native-default only -- NO 50%.
GSM8K_BUDGET_LABELS: Sequence[str] = cost.GSM8K_BUDGET_LABELS
GSMIC_BUDGET_LABELS: Sequence[str] = cost.GSMIC_BUDGET_LABELS
# BUDGET_LABELS is the GSM8K four; GSM-IC uses GSMIC_BUDGET_LABELS.
BUDGET_LABELS: Sequence[str] = GSM8K_BUDGET_LABELS

# Both conditions are always swept: the CLEAN condition is boundary case 1
# (nothing to select away) and the DISTRACTOR condition exercises boundary
# case 2 (an in-record distractor).
CONDITIONS: Sequence[str] = ("clean", "distractor")

DEFAULT_MODEL = "gpt-4o-mini"  # the reference base model; see cost.py's pricing caveat
DEFAULT_OPERATING_POINT = "np-2026-07-r1"
MAX_COMPLETION_TOKENS = 300  # generous ceiling for GSM chain-of-thought + final numeric answer
DEFAULT_MAX_WORKERS = 15  # conservative default concurrency for the per-cell completion calls


def default_budget_labels(mode: str) -> Sequence[str]:
    """The mode's default budget-label set (GSM-IC omits ``50pct``)."""
    return GSMIC_BUDGET_LABELS if mode == MODE_GSM_IC else GSM8K_BUDGET_LABELS


def allowed_budget_labels(mode: str) -> frozenset:
    return frozenset(default_budget_labels(mode))


def _cost_method_for_arm(arm_name: str) -> str:
    """Map an arm name to the method the budget/cost model resolves it as.

    ``full_context`` / ``needlepath`` / ``llmlingua2`` map to themselves. ``cpc``
    is another whole-content compression arm, so the budget/cost model (which has
    no cpc-specific rate) treats it identically to ``llmlingua2`` -- a documented
    equivalence for the budget-resolution and the conservative pre-flight
    estimate only; the arm itself still runs its own real compressor.
    """
    return "llmlingua2" if arm_name == "cpc" else arm_name


# --------------------------------------------------------------------------- #
# Budget resolution (single source of truth: the cost module)
# --------------------------------------------------------------------------- #


def resolve_budget_tokens(mode: str, method: str, budget_label: str, item: Any) -> int:
    """Resolve a ``(mode, method, budget_label)`` pair to a concrete per-item
    ``budget_tokens`` value for ``item``.

    Delegates to the cost module's resolvers so the budget the sweep applies and
    the budget the pre-flight cost estimate prices are computed by ONE piece of
    code (a hard invariant: the printed estimate must describe the run that
    actually happens). Percentage budgets are matched as a % of the item's
    QUESTION-content tokens (not the full prompt's); ``full_context``'s
    ``native_default`` is a documented, ignored placeholder (that arm never uses
    its budget argument).
    """
    cost_method = _cost_method_for_arm(method)
    if mode == MODE_GSM_IC:
        return cost.resolve_gsmic_budget_tokens(cost_method, budget_label, item)
    return cost.resolve_gsm8k_budget_tokens(cost_method, budget_label, item)


# --------------------------------------------------------------------------- #
# Condition grouping + model-prompt assembly
# --------------------------------------------------------------------------- #


def group_examples_by_condition(examples: Sequence[Any]) -> Dict[str, List[Any]]:
    """Group prompt examples / adapter items by ``.condition``.

    Works uniformly over GSM8K prompt examples and GSM-IC adapter items (both
    carry ``.condition``): only that attribute is touched.
    """
    grouped: Dict[str, List[Any]] = {c: [] for c in CONDITIONS}
    for example in examples:
        grouped.setdefault(example.condition, []).append(example)
    return grouped


def assemble_model_prompt(arm_name: str, rendered_context: str, fewshot: str) -> str:
    """Reassemble the model prompt from an arm's ``rendered_context``.

    Reproduces the reference control flow's three-way split exactly:

    - ``full_context``: the rendered context IS the whole assembled prompt
      (byte-identical to the raw prompt), sent verbatim -- the fixed few-shot is
      already inside it, so it is NOT re-prepended.
    - every other arm (``needlepath`` and the whole-content compression arms):
      the rendered context is the reduced QUESTION content, so the identical,
      fixed few-shot block is re-attached verbatim via the suite's single
      prompt-assembly convention (``f"{fewshot}\\n\\nQuestion: {q}\\nAnswer:"``).

    Keeping every method on the identical few-shot scaffolding is what makes this
    a matched protocol: no method gets a setup edge from how the prompt is built.
    """
    if arm_name == BASELINE_ARM:
        return rendered_context
    return _assemble_prompt(fewshot, rendered_context)


# --------------------------------------------------------------------------- #
# Per-item + per-cell result shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PreparedItem:
    """One item's selector output, ready for a single model completion.

    Carries everything the completion + row assembly needs, so the (sequential)
    selection phase and the (concurrent) completion phase are cleanly separable,
    matching the reference harness's two-phase structure (adapter-level context
    preparation first, model completions second).
    """

    item_id: int
    base_problem_id: Optional[int]
    condition: str
    gold_numeric: float
    budget_tokens: int
    prompt: str
    response: ContextResponse


@dataclass
class SweepCellResult:
    """Aggregates for one ``(arm x budget x condition)`` cell.

    ``em_rate``/``n`` are the MICRO (per-example exact-match) axis. For GSM-IC,
    ``macro_accuracy``/``n_base_problems``/``failed_base_problem_ids`` add the
    per-BASE-PROBLEM axis (a base problem counts as correct only if EVERY sampled
    variant of it was individually correct); they are ``None``/empty for GSM8K,
    which has no base-problem variant structure. Per the benchmark's convention,
    macro is a coarser secondary sanity check with a much smaller true N and a
    correspondingly wide CI -- never headlined over the micro rate.

    ``prep_errors`` (selector/budget failures) and ``completion_errors`` (real
    model-completion failures) are kept separate since they occur at different
    stages; the affected items are excluded from ``item_rows``/the aggregates but
    are never silently dropped.
    """

    mode: str
    arm: str
    budget_label: str
    condition: str
    n: int
    em_rate: float
    macro_accuracy: Optional[float]
    n_base_problems: Optional[int]
    failed_base_problem_ids: List[int]
    avg_reduction_ratio: Optional[float]
    fallback_rate: Optional[float]
    mean_engine_latency_ms: float
    mean_model_latency_ms: float
    item_rows: List[Dict[str, Any]] = field(default_factory=list)
    prep_errors: List[Dict[str, Any]] = field(default_factory=list)
    completion_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        """The per-cell summary row (no per-item rows)."""
        return {
            "mode": self.mode,
            "arm": self.arm,
            "budget_label": self.budget_label,
            "condition": self.condition,
            "n": self.n,
            "em_rate": self.em_rate,
            "macro_accuracy": self.macro_accuracy,
            "n_base_problems": self.n_base_problems,
            "failed_base_problem_ids": list(self.failed_base_problem_ids),
            "avg_reduction_ratio": self.avg_reduction_ratio,
            "fallback_rate": self.fallback_rate,
            "mean_engine_latency_ms": self.mean_engine_latency_ms,
            "mean_model_latency_ms": self.mean_model_latency_ms,
            "n_prep_errors": len(self.prep_errors),
            "n_completion_errors": len(self.completion_errors),
        }


def _error_row(
    *,
    mode: str,
    arm_name: str,
    budget_label: str,
    condition: str,
    item_id: Optional[int],
    stage: str,
    exc: Exception,
) -> Dict[str, Any]:
    """A compact per-item error row (type/message; never a full traceback)."""
    return {
        "mode": mode,
        "arm": arm_name,
        "budget_label": budget_label,
        "condition": condition,
        "item_id": item_id,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _build_item_row(
    *,
    mode: str,
    arm_name: str,
    budget_label: str,
    prepared: _PreparedItem,
    answer: str,
    correct: bool,
    model_latency_ms: float,
) -> Dict[str, Any]:
    """Assemble one per-item output row from the arm response + model result.

    Carries the same information the reference per-item row carried, read
    straight off ``ContextResponse`` (token deltas, fallback flag, budget,
    latency, selected ids, and the neutral safety subset when the arm produced
    one -- ``None`` for pure compression / full-context arms).
    """
    response = prepared.response
    safety = response.safety
    return {
        "mode": mode,
        "arm": arm_name,
        "budget_label": budget_label,
        "condition": prepared.condition,
        "item_id": prepared.item_id,
        "base_problem_id": prepared.base_problem_id,
        "gold_numeric": prepared.gold_numeric,
        "resolved_budget_tokens": prepared.budget_tokens,
        "answer": answer,
        "correct": correct,
        "model_latency_ms": model_latency_ms,
        # --- read off ContextResponse ---
        "tokens_before": response.tokens_before,
        "tokens_after": response.tokens_after,
        "tokens_saved": response.tokens_saved,
        "reduction_ratio": response.reduction_ratio,
        "records_available": response.records_available,
        "records_selected": response.records_selected,
        "fallback_used": response.fallback_used,
        "selection_error": response.selection_error,
        "engine_latency_ms": response.engine_latency_ms,
        "budget_tokens": response.budget_tokens,
        "attempted_budget_tokens": list(response.attempted_budget_tokens),
        "selected_record_ids": [s.record_id for s in response.selected],
        # --- neutral safety subset (None for pure compression / full context) ---
        "selection_safe": None if safety is None else safety.selection_safe,
        "fallback_reason": "" if safety is None else safety.fallback_reason,
        "coverage_score": None if safety is None else safety.coverage_score,
        "evidence_shape": "" if safety is None else str(safety.evidence_shape),
    }


def _mean(values: Sequence[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Cell execution (one arm x budget x condition)
# --------------------------------------------------------------------------- #


def run_sweep_cell(
    *,
    mode: str,
    arm_name: str,
    arm: ContextArm,
    budget_label: str,
    condition: str,
    items: Sequence[Any],
    client: Any,
    model: str,
    fewshot: str,
    operating_point: Optional[str],
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> SweepCellResult:
    """Run all ``items`` (a single condition's worth) through one arm at the
    resolved budget for ``(mode, arm, budget_label)``, then score EM.

    Two phases, mirroring the reference harness:

    * Selection (sequential): resolve the per-item budget, build the request,
      run ``arm.select``, and reassemble the model prompt. A single item raising
      here (malformed item, transient selector error) is recorded as a compact
      ``prep`` error row and skipped, never aborting the cell.
    * Completion (concurrent): one model completion per prepared item via a
      ``ThreadPoolExecutor``, scored with ``score_exact_match``. A single
      completion raising is recorded as a compact ``completion`` error row and
      skipped; ``em_rate``/``n`` are computed only over items actually scored.
    """
    prepared: List[_PreparedItem] = []
    prep_errors: List[Dict[str, Any]] = []
    for item in items:
        try:
            budget_tokens = resolve_budget_tokens(mode, arm_name, budget_label, item)
            request = build_gsm_request(
                item,
                arm_name=arm_name,
                budget=budget_tokens,
                operating_point=operating_point,
            )
            response = arm.select(request)
            prompt = assemble_model_prompt(arm_name, response.rendered_context, fewshot)
        except Exception as exc:  # noqa: BLE001 - isolate any per-item selection failure
            prep_errors.append(
                _error_row(
                    mode=mode,
                    arm_name=arm_name,
                    budget_label=budget_label,
                    condition=condition,
                    item_id=getattr(item, "item_id", None),
                    stage="prepare",
                    exc=exc,
                )
            )
            continue
        prepared.append(
            _PreparedItem(
                item_id=item.item_id,
                base_problem_id=getattr(item, "base_problem_id", None),
                condition=condition,
                gold_numeric=item.gold_numeric,
                budget_tokens=budget_tokens,
                prompt=prompt,
                response=response,
            )
        )

    def _score_one(prep: _PreparedItem):
        started_at = time.perf_counter()
        try:
            answer = call_openai(
                client, model=model, prompt=prep.prompt, max_tokens=max_completion_tokens
            )
        except Exception as exc:  # noqa: BLE001 - isolate any per-item completion failure
            return (
                "error",
                _error_row(
                    mode=mode,
                    arm_name=arm_name,
                    budget_label=budget_label,
                    condition=condition,
                    item_id=prep.item_id,
                    stage="completion",
                    exc=exc,
                ),
            )
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        correct = score_exact_match(answer, prep.gold_numeric)
        return (
            "ok",
            _build_item_row(
                mode=mode,
                arm_name=arm_name,
                budget_label=budget_label,
                prepared=prep,
                answer=answer,
                correct=correct,
                model_latency_ms=latency_ms,
            ),
        )

    item_rows: List[Dict[str, Any]] = []
    completion_errors: List[Dict[str, Any]] = []
    if prepared:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for status, payload in executor.map(_score_one, prepared):
                if status == "ok":
                    item_rows.append(payload)
                else:
                    completion_errors.append(payload)

    # Deterministic ordering (the concurrent completion phase does not preserve
    # submission order): sort scored rows by item_id so per-item output files are
    # byte-stable across runs with identical inputs.
    item_rows.sort(key=lambda r: r["item_id"])

    n = len(item_rows)
    em_rate = (sum(1 for r in item_rows if r["correct"]) / n) if n else 0.0
    avg_reduction = _mean([r["reduction_ratio"] for r in item_rows]) if n else None
    fallback = (sum(1 for r in item_rows if r["fallback_used"]) / n) if n else None

    macro_acc: Optional[float] = None
    n_base_problems: Optional[int] = None
    failed_base_problem_ids: List[int] = []
    if mode == MODE_GSM_IC and item_rows and all(r["base_problem_id"] is not None for r in item_rows):
        macro_result = macro_accuracy(
            [(int(r["base_problem_id"]), bool(r["correct"])) for r in item_rows]
        )
        macro_acc = macro_result.accuracy
        n_base_problems = macro_result.n_base_problems
        failed_base_problem_ids = list(macro_result.failed_base_problem_ids)

    return SweepCellResult(
        mode=mode,
        arm=arm_name,
        budget_label=budget_label,
        condition=condition,
        n=n,
        em_rate=em_rate,
        macro_accuracy=macro_acc,
        n_base_problems=n_base_problems,
        failed_base_problem_ids=failed_base_problem_ids,
        avg_reduction_ratio=avg_reduction,
        fallback_rate=fallback,
        mean_engine_latency_ms=_mean([r["engine_latency_ms"] for r in item_rows]),
        mean_model_latency_ms=_mean([r["model_latency_ms"] for r in item_rows]),
        item_rows=item_rows,
        prep_errors=prep_errors,
        completion_errors=completion_errors,
    )


# --------------------------------------------------------------------------- #
# Sweep matrix
# --------------------------------------------------------------------------- #


def run_sweep(
    *,
    mode: str,
    items_by_condition: Dict[str, Sequence[Any]],
    arm_objects: Dict[str, ContextArm],
    arms: Sequence[str],
    budget_labels: Sequence[str],
    client: Any,
    model: str,
    fewshot: str,
    operating_point: Optional[str],
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    cell_sink: Optional[Callable[[SweepCellResult], None]] = None,
) -> List[SweepCellResult]:
    """Run the full ``arm x condition x budget`` matrix (real arms + real model).

    For each cell it builds requests via ``build_gsm_request``, runs
    ``arm.select``, reassembles the model prompt, calls the model, scores EM, and
    aggregates (with ``macro_accuracy`` for GSM-IC). ``cell_sink``, when given, is
    invoked with each cell as soon as it completes -- the orchestration uses it
    for incremental per-cell persistence, so a mid-run failure loses at most the
    in-flight cell.

    Iteration order (arm -> condition -> budget) matches the reference sweep.
    """
    cells: List[SweepCellResult] = []
    for arm_name in arms:
        arm = arm_objects[arm_name]
        for condition, items in items_by_condition.items():
            for budget_label in budget_labels:
                cell = run_sweep_cell(
                    mode=mode,
                    arm_name=arm_name,
                    arm=arm,
                    budget_label=budget_label,
                    condition=condition,
                    items=items,
                    client=client,
                    model=model,
                    fewshot=fewshot,
                    operating_point=operating_point,
                    max_completion_tokens=max_completion_tokens,
                    max_workers=max_workers,
                )
                cells.append(cell)
                if cell_sink is not None:
                    cell_sink(cell)
    return cells


# --------------------------------------------------------------------------- #
# Arm construction (by name) -- mirrors the other suites' builders
# --------------------------------------------------------------------------- #


def build_arm(
    name: str,
    *,
    needlepath_url: Optional[str] = None,
    operating_point: Optional[str] = None,
    request_timeout_ms: int = 120000,
) -> ContextArm:
    """Construct an arm by name.

    ``full_context`` and ``needlepath`` are light and always available.
    ``llmlingua2`` and ``cpc`` carry heavy optional dependencies and are imported
    lazily so merely constructing the common arms never pulls in the model
    stacks. The hosted ``needlepath`` arm reads its bearer token from
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
# Data loading (per mode)
# --------------------------------------------------------------------------- #


def load_items_by_condition(mode: str, *, n: int, seed: int, fewshot: str):
    """Load ``n`` items for ``mode`` and return
    ``(items_by_condition, itemid_to_base_problem_id_or_None)``.

    - GSM8K: the first ``n`` ``test``-split items in dataset order
      (``load_gsm8k_test_items()[:n]`` -- the same seeded selection the cost
      estimate uses, so both agree on the item set), each expanded into CLEAN +
      DISTRACTOR prompt examples with the shared few-shot prefix.
    - GSM-IC: the real, balanced STRATIFIED sample (``n / <base problems>`` per
      base problem) when ``n`` is a positive multiple of the base-problem count;
      otherwise a plain first-``n`` slice (a documented smoke-scale fallback, the
      same decision the reference smoke path made -- a stratified sample requires
      per-base-problem balance a small ``n`` cannot satisfy). Each item is
      expanded into CLEAN + DISTRACTOR adapter items with its sentence-unit
      segmentation.
    """
    if mode == MODE_GSM_IC:
        try:
            items = sample_gsmic_2step(n=n, seed=seed)
        except ValueError:
            print(
                f"[gsm_ic] n={n} is not a positive multiple of the base-problem count; "
                "falling back to a plain first-n slice (smoke scale, unbalanced).",
                file=sys.stderr,
            )
            items = load_gsmic_2step_items()[:n]
        adapter_items = build_gsmic_adapter_items(items, fewshot)
        return group_examples_by_condition(adapter_items)

    gsm8k_items = load_gsm8k_test_items()[:n]
    examples = build_prompt_examples(gsm8k_items, fewshot, seed=seed)
    return group_examples_by_condition(examples)


# --------------------------------------------------------------------------- #
# Budget hard-stop (mirrors csbench.suites.ruler.harness.enforce_budget)
# --------------------------------------------------------------------------- #


def enforce_budget(
    *,
    mode: str,
    arms: Sequence[str],
    n: int,
    seed: int,
    budget_labels: Sequence[str],
    conditions: Sequence[str] = CONDITIONS,
) -> float:
    """Print the pre-run cost estimate and abort (before any API call) if the
    estimated spend would exceed the ``BENCH_BUDGET_USD`` env hard-stop.

    The estimate comes from ``csbench.suites.gsm.cost`` (real data + the same
    per-item budget math the sweep uses), so the printed figure describes the run
    that will actually happen. Each arm is priced as its budget/cost-equivalent
    method (``cpc`` as a compression arm), preserving the per-arm completion
    count. Returns the estimate. No env var set means no cap.
    """
    cost_methods = [_cost_method_for_arm(a) for a in arms]
    if mode == MODE_GSM_IC:
        estimate_obj = cost.estimate_gsmic_pilot_cost(
            n=n, seed=seed, methods=cost_methods, budget_labels=budget_labels, conditions=conditions
        )
        print(cost.format_gsmic_pilot_cost_report(estimate_obj), file=sys.stderr)
    else:
        estimate_obj = cost.estimate_pilot_cost(
            n=n, seed=seed, methods=cost_methods, budget_labels=budget_labels, conditions=conditions
        )
        print(cost.format_pilot_cost_report(estimate_obj), file=sys.stderr)

    estimate = estimate_obj.total_cost_usd
    cap_raw = os.environ.get("BENCH_BUDGET_USD")
    if cap_raw is not None and cap_raw.strip():
        cap = float(cap_raw)
        if estimate > cap:
            raise SystemExit(
                f"BENCH_BUDGET_USD hard-stop: estimated spend ${estimate:.4f} exceeds cap "
                f"${cap:.2f} (mode={mode}, arms={len(arms)} x budgets={len(budget_labels)} x "
                f"conditions={len(conditions)} x n={n}). Lower --n / --budgets / --arms or "
                "raise the cap."
            )
    return estimate


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _cell_items_path(out_root: Path, cell: SweepCellResult) -> Path:
    return Path(out_root) / "items" / cell.condition / cell.arm / f"{cell.budget_label}.jsonl"


def _write_cell_items(out_root: Path, cell: SweepCellResult) -> Path:
    path = _cell_items_path(out_root, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in cell.item_rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True))
            handle.write("\n")
    return path


def write_summary(out_root: Path, mode: str, cells: Sequence[SweepCellResult]) -> Path:
    """Per-cell summary rows + a combined per-(arm, budget) rollup, as JSON + MD."""
    summaries = [cell.to_summary() for cell in cells]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "note": (
            "GSM boundary suite -- a DIAGNOSTIC, not a flagship result. GSM8K clean is "
            "boundary case 1 (nothing to select away, so selection is a no-op by design); "
            "GSM-IC's inline distractor is boundary case 2 (a distractor sentence inside "
            "the single record a solver must keep, outside the record-level selection "
            "regime). All arms share identical few-shot scaffolding and one model. "
            "macro_accuracy (GSM-IC only) is a coarse secondary check with a small true N; "
            "it is never headlined over the micro em_rate."
        ),
        "cells": summaries,
    }
    json_path = Path(out_root) / "summary.json"
    json_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

    lines = [
        f"# GSM boundary-suite summary ({mode})",
        "",
        "Diagnostic, not a flagship result. See README.md for the two disclosed boundary cases.",
        "",
        "| Arm | Budget | Condition | N | EM | Macro | Avg reduction | Fallback rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        macro = "" if s["macro_accuracy"] is None else f"{s['macro_accuracy']:.4f}"
        reduction = "" if s["avg_reduction_ratio"] is None else f"{s['avg_reduction_ratio'] * 100.0:.2f}%"
        fallback = "" if s["fallback_rate"] is None else f"{s['fallback_rate'] * 100.0:.2f}%"
        lines.append(
            f"| {s['arm']} | {s['budget_label']} | {s['condition']} | {s['n']} | "
            f"{s['em_rate']:.4f} | {macro} | {reduction} | {fallback} |"
        )
    md_path = Path(out_root) / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #


def run(
    *,
    mode: str,
    arms: Sequence[str],
    n: int,
    seed: int,
    budget_labels: Sequence[str],
    model: str,
    operating_point: Optional[str],
    needlepath_url: Optional[str],
    out_root: Path,
    run_name: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    request_timeout_ms: int = 120000,
    estimated_spend_usd: float = 0.0,
) -> Dict[str, Any]:
    """Load data, run the sweep, and write every artifact. Returns a manifest."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fewshot = build_fewshot_prompt()
    items_by_condition = load_items_by_condition(mode, n=n, seed=seed, fewshot=fewshot)

    arm_objects = {
        name: build_arm(
            name,
            needlepath_url=needlepath_url,
            operating_point=operating_point,
            request_timeout_ms=request_timeout_ms,
        )
        for name in arms
    }
    client = build_openai_client()

    config = {
        "run_name": run_name,
        "mode": mode,
        "arms": list(arms),
        "budget_labels": list(budget_labels),
        "conditions": list(CONDITIONS),
        "n": n,
        "seed": seed,
        "model": model,
        "operating_point": operating_point or "",
        "needlepath_url": needlepath_url or "",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "concurrency": max_workers,
        "estimated_spend_usd": estimated_spend_usd,
        "out_root": str(out_root),
        "command": " ".join(sys.argv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "run_config.json").write_text(
        json.dumps(json_safe(config), indent=2), encoding="utf-8"
    )
    (out_root / "git_provenance.json").write_text(
        json.dumps(json_safe(probe_git_state()), indent=2), encoding="utf-8"
    )

    output_files: List[Path] = []

    def _sink(cell: SweepCellResult) -> None:
        # Incremental per-cell persistence: write this cell's per-item JSONL as
        # soon as it completes, so a mid-run failure loses at most one cell.
        path = _write_cell_items(out_root, cell)
        output_files.append(path)
        print(
            f"[{mode}] {cell.arm} / {cell.budget_label} / {cell.condition}: "
            f"n={cell.n} em={cell.em_rate:.3f}"
            + ("" if cell.macro_accuracy is None else f" macro={cell.macro_accuracy:.3f}")
            + (
                ""
                if cell.fallback_rate is None
                else f" fallback={cell.fallback_rate * 100.0:.1f}%"
            ),
            file=sys.stderr,
        )

    cells = run_sweep(
        mode=mode,
        items_by_condition=items_by_condition,
        arm_objects=arm_objects,
        arms=arms,
        budget_labels=budget_labels,
        client=client,
        model=model,
        fewshot=fewshot,
        operating_point=operating_point,
        max_workers=max_workers,
        cell_sink=_sink,
    )

    write_summary(out_root, mode, cells)
    manifest_path = write_output_manifest(
        out_root / "manifest.sha256", output_files, base_dir=out_root
    )

    return {
        "out_root": str(out_root),
        "run_name": run_name,
        "mode": mode,
        "arms": list(arms),
        "budget_labels": list(budget_labels),
        "n": n,
        "n_cells": len(cells),
        "n_output_files": len(output_files),
        "summary": str(out_root / "summary.json"),
        "output_manifest": str(manifest_path),
        "estimated_spend_usd": estimated_spend_usd,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_csv(value: str) -> tuple:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GSM boundary-suite matched-budget sweep (DIAGNOSTIC, not flagship): run GSM8K "
            "or GSM-IC through context-selection arms end-to-end with a BENCH_BUDGET_USD "
            "pre-flight hard stop."
        )
    )
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=MODE_GSM8K,
                        help="gsm8k (clean single-doc math) or gsm_ic (inline distractor)")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="comma-separated arm names: full_context,needlepath,llmlingua2,cpc")
    parser.add_argument("--n", type=int, default=None,
                        help="item count; default per mode (gsm8k: "
                        f"{cost.DEFAULT_ESTIMATE_N}; gsm_ic: {cost.DEFAULT_GSMIC_ESTIMATE_N}). "
                        "GSM8K: first-n test items; GSM-IC: stratified sample size.")
    parser.add_argument("--budgets", default="",
                        help="comma-separated budget labels; default per mode "
                        "(gsm8k: 25pct,50pct,75pct,native_default; gsm_ic: 25pct,75pct,native_default)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--needlepath-url", default=None,
                        help="base URL of the hosted Needlepath endpoint (required for the needlepath arm)")
    parser.add_argument("--operating-point", default=DEFAULT_OPERATING_POINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("runs/gsm"))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"max concurrent in-flight model completions per cell (default {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--request-timeout-ms", type=int, default=120000)
    parser.add_argument("--run-name",
                        default=f"gsm_{datetime.now().strftime('%Y%m%d')}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    mode = args.mode
    arms = _parse_csv(args.arms)
    unknown_arms = [a for a in arms if a not in SUPPORTED_ARMS]
    if unknown_arms:
        raise SystemExit(f"Unknown arms: {', '.join(sorted(unknown_arms))}")
    if not arms:
        raise SystemExit("--arms must name at least one arm")

    n = args.n
    if n is None:
        n = cost.DEFAULT_GSMIC_ESTIMATE_N if mode == MODE_GSM_IC else cost.DEFAULT_ESTIMATE_N
    if n < 1:
        raise SystemExit("--n must be >= 1")
    if "needlepath" in arms and not args.needlepath_url:
        raise SystemExit("the 'needlepath' arm requires --needlepath-url")

    budget_labels = _parse_csv(args.budgets) if args.budgets else tuple(default_budget_labels(mode))
    allowed = allowed_budget_labels(mode)
    unknown_budgets = [b for b in budget_labels if b not in allowed]
    if unknown_budgets:
        raise SystemExit(
            f"Unknown budget label(s) for mode {mode!r}: {', '.join(unknown_budgets)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    # Estimate spend and abort BEFORE constructing any client or making any call.
    estimate = enforce_budget(
        mode=mode, arms=arms, n=n, seed=args.seed, budget_labels=budget_labels
    )
    print(f"estimated spend: ${estimate:.4f}", file=sys.stderr)

    result = run(
        mode=mode,
        arms=arms,
        n=n,
        seed=args.seed,
        budget_labels=budget_labels,
        model=args.model,
        operating_point=args.operating_point,
        needlepath_url=args.needlepath_url,
        out_root=args.out,
        run_name=args.run_name,
        max_workers=args.concurrency,
        request_timeout_ms=args.request_timeout_ms,
        estimated_spend_usd=estimate,
    )
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
