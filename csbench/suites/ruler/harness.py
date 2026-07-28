"""RULER suite driver: run arms end-to-end, score, aggregate, pin provenance.

This is the port of the reference RULER benchmark's control flow. The one
structural change from the internal version: the internal per-item engine call
``build_selection(...)`` is replaced by the uniform arm contract
``arm.select(request) -> ContextResponse``. Everything downstream reads the same
fields off ``ContextResponse`` that the internal per-item row dict carried, and
the model prompt is assembled by the byte-faithful ``build_ruler_prompt``.

For each RULER example the driver:

  1. splits the official prompt into ``(context, question)``;
  2. builds a per-arm ``ContextRequest`` (full-context passthrough, the hosted
     Needlepath selector over chunked records, or a whole-context compression
     arm);
  3. runs ``arm.select(request)``;
  4. assembles ``build_ruler_prompt(response.rendered_context, question,
     answer_prefix)`` and calls the model;
  5. scores with ``score_ruler_answer`` under the task's match type;
  6. writes a per-item row (JSONL) carrying the same columns the internal row
     carried, read straight off ``ContextResponse``.

It then aggregates per ``(arm x task x length)`` (accuracy, fallback rate,
token reduction, and, against the full-context baseline, a paired McNemar test
and a bootstrap CI on the exact-match delta), writes a matrix + combined
summary, and emits a sha256 manifest over every per-item output file so a run is
independently verifiable via ``tools/verify_manifests.py``.

Model, scorer, and seed knobs match the published run. Arms that carry heavy
optional dependencies (LLMLingua-2, CPC) are imported lazily, only when named.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from csbench.arms import FullContextArm, NeedlepathArm
from csbench.arms.base import ContextArm
from csbench.contracts import (
    AdaptiveBudget,
    BudgetSpec,
    ContextRecord,
    ContextRequest,
    ContextResponse,
    TaskSpec,
)
from csbench.model_client import build_gemini_client, call_gemini, resolve_model
from csbench.provenance import (
    build_trusted_benchmark_manifest,
    enforce_clean_worktree,
    json_safe,
    write_output_manifest,
)
from csbench.stats import bootstrap_ci_delta, mcnemar_test
from csbench.suites.ruler.pairing import (
    ItemIdentityError,
    assert_item_identity,
    forbid_index_keying,
    pair_rows,
)
from csbench.suites.ruler.data import (
    OfficialRulerExample,
    chunk_text,
    load_examples,
    match_type_for_task,
    score_ruler_answer,
    selection_keywords_for_example,
    split_official_prompt,
)
from csbench.suites.ruler.prompts import baseline_prompt, compression_prompt, needlepath_prompt

# --------------------------------------------------------------------------- #
# Defaults (match the published run)
# --------------------------------------------------------------------------- #

DEFAULT_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
DEFAULT_TASKS_ARG = ",".join(DEFAULT_TASKS)
DEFAULT_LENGTHS = ("8k", "16k")
DEFAULT_ARMS = ("full_context", "needlepath")

BASELINE_ARM = "full_context"

# Selection/budget operating point (mirrors the reference adaptive adapter).
DEFAULT_MAX_SELECTED_TOKENS = 4096
DEFAULT_MAX_SELECTED_RECORDS = 18
DEFAULT_INITIAL_BUDGET = 4096
DEFAULT_BUDGET_LADDER = (8192,)
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_OPERATING_POINT = "np-2026-07-r1"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 10000

# Fields the hosted Needlepath selector reproduces from the reference adapter.
RULER_TOOL_NAME = "lookup_ruler_context"
RULER_SOURCE = "official_nvidia_ruler"

# EM binarization threshold: a RULER score of 1.0 means every reference matched.
_EM_EPS = 1e-9

# Conservative default $/1k-token blended rate for the pre-run budget estimate.
# Deliberately high so the hard-stop over-estimates and never under-bills a run.
# Override via BENCH_USD_PER_1K_TOKENS. This is a safety guard, not billing.
DEFAULT_USD_PER_1K_TOKENS = 0.015


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return fmean(vals) if vals else 0.0


def _length_tokens(length_label: str) -> int:
    """Best-effort token count for a length label like ``8k``/``16k``/``32768``."""
    label = str(length_label).strip().lower()
    if label.endswith("k"):
        return int(float(label[:-1]) * 1024)
    return int(label)


def _length_data_root(data_root: Path, length_label: str) -> Path:
    """Per-length dataset root: ``<data_root>/<length>/<task>/test.jsonl``."""
    return Path(data_root) / length_label


# --------------------------------------------------------------------------- #
# Arm construction (by name)
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
    model stacks.
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
    if name == "compresr":
        from csbench.arms.compresr import CompresrArm  # optional dep; reads COMPRESR_API_KEY

        return CompresrArm(model="latte_v1")  # default operating point, no tuning
    raise ValueError(f"unknown arm: {name!r}")


def input_sha256(example: "OfficialRulerExample") -> str:
    """SHA-256 of the generated input text for one example.

    The 2026-07-17 deposit recorded only derived quantities (token counts,
    lengths), so when the dataset was regenerated ten days later, byte-identity
    of the haystacks could not be verified after the fact -- only inferred from
    agreeing fingerprints. Persisting this hash makes a later regeneration
    checkable byte-for-byte instead.
    """
    return hashlib.sha256((example.input_text or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Per-arm request construction
# --------------------------------------------------------------------------- #


def _budget_spec(operating_point: str | None) -> BudgetSpec:
    """The shared operating point.

    Adaptive selection with the reference ladder for the hosted selector; local
    arms read only ``max_context_tokens`` off it. Built once per run.
    """
    return BudgetSpec(
        max_context_tokens=DEFAULT_MAX_SELECTED_TOKENS,
        operating_point=operating_point,
        max_records=DEFAULT_MAX_SELECTED_RECORDS,
        mode="adaptive",
        adaptive=AdaptiveBudget(
            initial_tokens=DEFAULT_INITIAL_BUDGET,
            escalation_tokens=list(DEFAULT_BUDGET_LADDER),
            allow_full_context_fallback=True,
        ),
        require_evidence_coverage=True,
    )


def build_request(
    arm_name: str,
    example: OfficialRulerExample,
    *,
    context: str,
    question: str,
    budget: BudgetSpec,
) -> ContextRequest:
    """Build the ``ContextRequest`` for one arm and one example.

    - ``full_context`` / ``llmlingua2`` / ``cpc``: ONE untitled record holding
      the whole context, so the full-context arm's rendered context is
      byte-identical to the raw context and the compression arms compress the
      whole context.
    - ``needlepath``: the context is chunked (``chunk_text``) into one
      ``external_data`` record per chunk (titled, sourced, importance 1.0, and
      tagged/keyworded exactly as the reference recorded them), and the hosted
      selector sub-selects.

    The task prompt reproduces the reference selection input
    ``f"{question}\\n{answer_prefix}"`` (the answer prefix carries no gold
    answer and is already reflected in the selection keywords), and the
    selection keywords are ``selection_keywords_for_example`` with gold answers
    excluded. ``render_format='hybrid'`` mirrors the reference context format.
    """
    keywords = list(selection_keywords_for_example(example, question))
    task = TaskSpec(
        prompt=f"{question}\n{example.answer_prefix}",
        tool_name=RULER_TOOL_NAME,
        keywords=keywords,
        tags=["ruler", example.task, "lookup", "long_context"],
    )
    request_id = f"official-ruler:{example.task}:{example.index}:{arm_name}"

    if arm_name == "needlepath":
        chunks = chunk_text(context)
        records = [
            ContextRecord(
                text=chunk,
                kind="external_data",
                id=f"{example.task}-{example.index}-chunk-{idx}",
                source=RULER_SOURCE,
                title=f"Official RULER context chunk {idx + 1}",
                importance=1.0,
                # No per-record keywords: the reference recorded chunks with no
                # explicit keywords (the engine infers them from the record
                # text). Selection keywords live only on the task signature.
                tags=["ruler", example.task, "official_context", "lookup", "long_context"],
                attributes={
                    "chunk_index": idx,
                    "ruler_task": example.task,
                    "ruler_index": example.index,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
    elif arm_name == BASELINE_ARM:
        # Control arm: the whole raw input_text as one record, so the rendered
        # context is byte-identical to the reference baseline input (which used
        # the un-split input_text, not the context/question split).
        records = [ContextRecord(text=example.input_text, kind="external_data")]
    else:
        # Compression arms (llmlingua2 / cpc): the split context as one record
        # (they compress the context; the question is re-attached at prompt time).
        records = [ContextRecord(text=context, kind="external_data")]

    return ContextRequest(
        request_id=request_id,
        records=records,
        task=task,
        budget=budget,
        render=True,
        render_format="hybrid",
        return_per_record=True,
    )


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #


def build_row(
    *,
    length: str,
    arm_name: str,
    position: int,
    example: OfficialRulerExample,
    response: ContextResponse,
    model: Mapping[str, Any],
    score: float,
    operating_point: str | None,
) -> dict[str, Any]:
    """Assemble one per-item row from the arm response + model result.

    Carries the same information the internal row dict carried, read straight
    off ``ContextResponse`` (token deltas, fallback flag, budget, latency,
    selected ids, and the neutral safety subset when the arm produced one).
    """
    match_type = match_type_for_task(example.task)
    em_correct = bool(score >= 1.0 - _EM_EPS)
    safety = response.safety
    return {
        "length": length,
        "task": example.task,
        "position": position,
        "index": example.index,
        # Byte-level provenance of the generated input. Recorded so a future
        # regeneration is checkable byte-for-byte rather than inferred from
        # derived quantities; see input_sha256().
        "input_sha256": input_sha256(example),
        "example_length": example.length,
        "arm": arm_name,
        "operating_point": operating_point or "",
        "policy_version": response.policy_version,
        "match_type": match_type,
        "expected_answer": list(example.expected_answer),
        "answer": model.get("content", ""),
        "score": score,
        "correct": em_correct,
        "input_tokens": model.get("input_tokens", 0),
        "output_tokens": model.get("output_tokens", 0),
        "model_latency_ms": model.get("latency_ms", 0.0),
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
        "format_metrics": dict(response.format_metrics),
    }


def run_item(
    arm: ContextArm,
    arm_name: str,
    example: OfficialRulerExample,
    *,
    position: int,
    client: Any,
    model: str,
    max_output_tokens: int,
    budget: BudgetSpec,
    operating_point: str | None,
) -> dict[str, Any]:
    """Run one (arm, example): select -> prompt -> model -> score -> row."""
    context, question = split_official_prompt(example.input_text)
    request = build_request(
        arm_name, example, context=context, question=question, budget=budget
    )
    response = arm.select(request)
    # Each arm's rendered context flows into its own byte-faithful template.
    if arm_name == BASELINE_ARM:
        prompt = baseline_prompt(response.rendered_context, example.answer_prefix)
    elif arm_name == "needlepath":
        prompt = needlepath_prompt(
            response.rendered_context,
            question,
            example.answer_prefix,
            fallback_used=response.fallback_used,
            input_text=example.input_text,
        )
    else:  # compression arms: llmlingua2 / cpc
        prompt = compression_prompt(response.rendered_context, question, example.answer_prefix)
    model_result = call_gemini(
        client, model=model, prompt=prompt, max_output_tokens=max_output_tokens
    )
    score = score_ruler_answer(
        model_result["content"], example.expected_answer, match_type_for_task(example.task)
    )
    return build_row(
        length="",  # filled by the caller which knows the length label
        arm_name=arm_name,
        position=position,
        example=example,
        response=response,
        model=model_result,
        score=score,
        operating_point=operating_point,
    )


# --------------------------------------------------------------------------- #
# Cell execution (one length x task x arm)
# --------------------------------------------------------------------------- #


def _cell_path(out_root: Path, length: str, arm_name: str, task: str) -> Path:
    return Path(out_root) / "items" / length / arm_name / f"{task}.jsonl"


def _load_existing_rows(path: Path, *, expected_n: int) -> dict[int, dict[str, Any]]:
    """Resume cache keyed by ``position``.

    This was previously keyed by RULER's ``index``, which is NOT unique within a
    task. On resume, several positions sharing an index received the SAME row
    object, which was then mutated (``position`` last-write-wins) and written
    once per position, so byte-identical copies replaced the evaluations they
    overwrote. 396 of 7800 rows were destroyed that way in the published
    replication. See ``csbench.suites.ruler.pairing``.
    """
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            position = int(row["position"])
        except (KeyError, TypeError, ValueError):
            continue
        if position in rows:
            raise ItemIdentityError(
                f"{path}: duplicate position {position} in an existing cell file. "
                "This file was written by the pre-fix resume path and has lost "
                "evaluations; it cannot be safely resumed. Re-run the cell with "
                "--rerun."
            )
        rows[position] = row
    # `expected_n` is the caller's example count. Inferring the bound from the
    # keys themselves would let an index-keyed cache define its own bound and
    # pass -- the precise mistake this tripwire exists to catch.
    forbid_index_keying(rows.keys(), source=f"{path} resume cache", n=expected_n)
    return rows


def _write_cell(path: Path, rows: Sequence[dict[str, Any]], *, expected_n: int) -> None:
    # Write-time item identity. This is where the 2026-07-17 defect should have
    # been caught: the cell had 100 rows but fewer than 100 distinct positions,
    # and nothing objected.
    # `expected_n` is the planned example count. Using len(rows) would let the
    # guard define its own expectation: a truncated prefix of positions 1..99
    # would be "complete" because 99 rows is 99 positions.
    assert_item_identity(rows, source=str(path), expected_n=expected_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True))
            handle.write("\n")


def run_cell(
    *,
    arm: ContextArm,
    arm_name: str,
    length: str,
    task: str,
    examples: Sequence[OfficialRulerExample],
    out_root: Path,
    client: Any,
    model: str,
    max_output_tokens: int,
    budget: BudgetSpec,
    operating_point: str | None,
    rerun: bool,
    sleep_seconds: float = 0.0,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run every example for one (length, task, arm) cell; persist the JSONL.

    Resumes by POSITION, cross-checking that the cached row belongs to the same
    underlying example. Returns the cell path and the ordered rows (by position).
    """
    path = _cell_path(out_root, length, arm_name, task)
    existing = {} if rerun else _load_existing_rows(path, expected_n=len(examples))

    rows_by_position: dict[int, dict[str, Any]] = {}
    for position, example in enumerate(examples, 1):
        prior = existing.get(position)
        if prior is not None and "score" in prior:
            if int(prior.get("index", -1)) != int(example.index):
                raise ItemIdentityError(
                    f"{path}: cached row at position {position} carries index "
                    f"{prior.get('index')} but the dataset has {example.index}. The "
                    "cached run used different data; re-run the cell with --rerun."
                )
            prior["length"] = length
            rows_by_position[position] = dict(prior)
            continue
        row = run_item(
            arm,
            arm_name,
            example,
            position=position,
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            budget=budget,
            operating_point=operating_point,
        )
        row["length"] = length
        rows_by_position[position] = row
        if sleep_seconds:
            time.sleep(sleep_seconds)

    rows = [rows_by_position[p] for p in sorted(rows_by_position)]
    _write_cell(path, rows, expected_n=len(examples))
    return path, rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def summarize_cell(length: str, task: str, arm_name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one (length, task, arm) cell of rows."""
    n = len(rows)
    coverage = [r["coverage_score"] for r in rows if r.get("coverage_score") is not None]
    return {
        "length": length,
        "task": task,
        "arm": arm_name,
        "n": n,
        # Primary accuracy is the mean RULER score (fractional-match aware),
        # matching the reference summary; em_accuracy is the binary all-match
        # rate used for the paired significance tests.
        "accuracy": _mean(float(r.get("score", 0.0)) for r in rows),
        "em_accuracy": _mean(1.0 if r.get("correct") else 0.0 for r in rows),
        "fallback_count": sum(1 for r in rows if r.get("fallback_used")),
        "avg_reduction_ratio": _mean(float(r.get("reduction_ratio", 0.0)) for r in rows),
        "avg_tokens_before": _mean(float(r.get("tokens_before", 0.0)) for r in rows),
        "avg_tokens_after": _mean(float(r.get("tokens_after", 0.0)) for r in rows),
        "avg_input_tokens": _mean(float(r.get("input_tokens", 0.0)) for r in rows),
        "avg_output_tokens": _mean(float(r.get("output_tokens", 0.0)) for r in rows),
        "avg_engine_latency_ms": _mean(float(r.get("engine_latency_ms", 0.0)) for r in rows),
        "avg_model_latency_ms": _mean(float(r.get("model_latency_ms", 0.0)) for r in rows),
        "avg_coverage_score": _mean(coverage) if coverage else None,
    }


def compare_to_baseline(
    arm_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any] | None:
    """Paired McNemar + bootstrap CI on the EM-correctness delta (arm minus
    baseline), aligned by ``position`` with an index identity assertion.

    Returns ``None`` for the baseline itself or when there is no shared item.
    """
    if not arm_rows or not baseline_rows:
        return None
    # Position-based pairing with a hard identity assertion. Never an
    # intersection: a shrinking paired count is a defect to surface, not a
    # population to quietly compute on.
    common, arm_aligned, base_aligned = pair_rows(
        arm_rows, baseline_rows, arm_source="arm", baseline_source="baseline"
    )
    arm_correct = [bool(r.get("correct")) for r in arm_aligned]
    base_correct = [bool(r.get("correct")) for r in base_aligned]

    mc = mcnemar_test(arm_correct, base_correct)
    boot = bootstrap_ci_delta(
        arm_correct, base_correct, n_resamples=n_resamples, seed=seed
    )
    # Full-prompt input-token reduction vs the baseline (economic view).
    arm_input = _mean(float(r.get("input_tokens", 0.0)) for r in arm_rows)
    base_input = _mean(float(r.get("input_tokens", 0.0)) for r in baseline_rows)
    final_input_reduction_pct = (
        (base_input - arm_input) / base_input * 100.0 if base_input else 0.0
    )
    return {
        "n_paired": len(common),
        "em_delta": boot.observed_delta,
        "em_delta_ci_low": boot.ci_low,
        "em_delta_ci_high": boot.ci_high,
        "bootstrap_confidence": boot.confidence,
        "bootstrap_resamples": boot.n_resamples,
        "mcnemar_b": mc.b,
        "mcnemar_c": mc.c,
        "mcnemar_statistic": mc.statistic,
        "mcnemar_p_value": mc.p_value,
        "mcnemar_method": mc.method,
        "final_input_reduction_pct": final_input_reduction_pct,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def write_matrix(out_root: Path, config: Mapping[str, Any], cells: Sequence[dict[str, Any]]) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "note": (
            "Official RULER-generated JSONL scored with RULER-style string matching. "
            "Every arm's context is wrapped in the identical prompt template; the "
            "full-context arm is the matched baseline. Selection/compression arms are "
            "adapters and are not part of the official RULER leaderboard harness. "
            "Expected answers are excluded from selection keywords."
        ),
        "cells": list(cells),
    }
    json_path = Path(out_root) / "matrix.json"
    json_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

    lines = [
        "# RULER matrix (matched-protocol, full-context baseline)",
        "",
        "| Length | Task | Arm | N | Accuracy | EM | Avg reduction | Fallbacks | EM delta vs full | McNemar p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        cmp = cell.get("vs_baseline")
        em_delta = f"{cmp['em_delta']:+.3f}" if cmp else ""
        mp = f"{cmp['mcnemar_p_value']:.3f}" if cmp else ""
        lines.append(
            f"| {cell['length']} | {cell['task']} | {cell['arm']} | {cell['n']} | "
            f"{cell['accuracy']:.3f} | {cell['em_accuracy']:.3f} | "
            f"{cell['avg_reduction_ratio'] * 100.0:.2f}% | {cell['fallback_count']} | "
            f"{em_delta} | {mp} |"
        )
    md_path = Path(out_root) / "matrix.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def write_combined_summary(out_root: Path, cells: Sequence[dict[str, Any]]) -> Path:
    """Per (arm, length) rollups across tasks, plus artifact completeness."""
    rollup: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["arm"], cell["length"])
        agg = rollup.setdefault(
            key,
            {"arm": cell["arm"], "length": cell["length"], "tasks": 0, "n": 0,
             "accuracy_sum": 0.0, "em_sum": 0.0, "fallbacks": 0, "reduction_sum": 0.0},
        )
        agg["tasks"] += 1
        agg["n"] += cell["n"]
        agg["accuracy_sum"] += cell["accuracy"]
        agg["em_sum"] += cell["em_accuracy"]
        agg["fallbacks"] += cell["fallback_count"]
        agg["reduction_sum"] += cell["avg_reduction_ratio"]

    rollups = []
    for agg in rollup.values():
        t = max(1, agg["tasks"])
        rollups.append(
            {
                "arm": agg["arm"],
                "length": agg["length"],
                "tasks": agg["tasks"],
                "total_items": agg["n"],
                "macro_over_tasks_accuracy": agg["accuracy_sum"] / t,
                "macro_over_tasks_em_accuracy": agg["em_sum"] / t,
                "avg_reduction_ratio": agg["reduction_sum"] / t,
                "fallbacks": agg["fallbacks"],
            }
        )
    rollups.sort(key=lambda r: (r["length"], r["arm"]))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "macro_accuracy_note": (
            "RULER examples have no base-problem variant structure, so per-item "
            "macro accuracy (in the GSM-IC sense) is not applicable; "
            "'macro_over_tasks_*' here is a simple task-averaged accuracy across "
            "the suite, reported as a secondary sanity check only."
        ),
        "rollups": rollups,
    }
    json_path = Path(out_root) / "combined_summary.json"
    json_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

    lines = [
        "# RULER combined summary",
        "",
        "| Length | Arm | Tasks | Items | Task-avg accuracy | Task-avg EM | Avg reduction | Fallbacks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rollups:
        lines.append(
            f"| {r['length']} | {r['arm']} | {r['tasks']} | {r['total_items']} | "
            f"{r['macro_over_tasks_accuracy']:.4f} | {r['macro_over_tasks_em_accuracy']:.4f} | "
            f"{r['avg_reduction_ratio'] * 100.0:.2f}% | {r['fallbacks']} |"
        )
    md_path = Path(out_root) / "combined_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


# --------------------------------------------------------------------------- #
# Budget hard-stop
# --------------------------------------------------------------------------- #


def estimate_cost_usd(
    *,
    arms: Sequence[str],
    tasks: Sequence[str],
    lengths: Sequence[str],
    n: int,
    max_output_tokens: int,
    usd_per_1k: float,
) -> float:
    """Conservative upper-bound cost estimate for the whole run.

    Assumes every model call sends the full context (the length label's token
    count) plus ``max_output_tokens`` of output, an over-estimate, since
    selection/compression arms send fewer input tokens. Deliberately pessimistic
    so the hard-stop errs toward aborting rather than overspending.
    """
    total_tokens = 0
    for length in lengths:
        per_call = _length_tokens(length) + max_output_tokens
        total_tokens += len(arms) * len(tasks) * n * per_call
    return total_tokens / 1000.0 * usd_per_1k


def enforce_budget(
    *,
    arms: Sequence[str],
    tasks: Sequence[str],
    lengths: Sequence[str],
    n: int,
    max_output_tokens: int,
) -> float:
    """Estimate spend and abort (before any API call) if it would exceed the
    ``BENCH_BUDGET_USD`` env hard-stop. Returns the estimate. No env var set
    means no cap."""
    usd_per_1k = float(os.environ.get("BENCH_USD_PER_1K_TOKENS", DEFAULT_USD_PER_1K_TOKENS))
    estimate = estimate_cost_usd(
        arms=arms,
        tasks=tasks,
        lengths=lengths,
        n=n,
        max_output_tokens=max_output_tokens,
        usd_per_1k=usd_per_1k,
    )
    cap_raw = os.environ.get("BENCH_BUDGET_USD")
    if cap_raw is not None and cap_raw.strip():
        cap = float(cap_raw)
        if estimate > cap:
            raise SystemExit(
                f"BENCH_BUDGET_USD hard-stop: estimated spend ${estimate:.2f} "
                f"exceeds cap ${cap:.2f} "
                f"(arms={len(arms)} x tasks={len(tasks)} x lengths={len(lengths)} x n={n}, "
                f"rate ${usd_per_1k:.4f}/1k tok). Lower --n / --tasks / --arms or raise the cap."
            )
    return estimate


# --------------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------------- #


def run_suite(
    *,
    arms: Sequence[str],
    tasks: Sequence[str],
    lengths: Sequence[str],
    n: int,
    data_root: Path,
    out_root: Path,
    client: Any,
    model: str,
    max_output_tokens: int,
    operating_point: str | None,
    needlepath_url: str | None,
    seed: int,
    bootstrap_resamples: int,
    run_name: str,
    rerun: bool = False,
    request_timeout_ms: int = 120000,
) -> dict[str, Any]:
    """Run every (length, task, arm), aggregate, and write all artifacts."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    budget = _budget_spec(operating_point)
    arm_objects = {
        name: build_arm(
            name,
            needlepath_url=needlepath_url,
            operating_point=operating_point,
            request_timeout_ms=request_timeout_ms,
        )
        for name in arms
    }

    config = {
        "run_name": run_name,
        "arms": list(arms),
        "tasks": list(tasks),
        "lengths": list(lengths),
        "n": n,
        "model": model,
        "operating_point": operating_point or "",
        "needlepath_url": needlepath_url or "",
        "max_output_tokens": max_output_tokens,
        "max_selected_tokens": DEFAULT_MAX_SELECTED_TOKENS,
        "max_selected_records": DEFAULT_MAX_SELECTED_RECORDS,
        "initial_budget": DEFAULT_INITIAL_BUDGET,
        "budget_ladder": list(DEFAULT_BUDGET_LADDER),
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "command": " ".join(sys.argv),
    }
    (out_root / "run_config.json").write_text(
        json.dumps(json_safe(config), indent=2), encoding="utf-8"
    )

    # Input provenance (dataset sha256s + git snapshot).
    context_roots = {length: _length_data_root(data_root, length) for length in lengths}
    manifest = build_trusted_benchmark_manifest(
        context_roots=context_roots,
        tasks=tasks,
        run_name=run_name,
        command=" ".join(sys.argv),
    )
    (out_root / "trusted_benchmark_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2), encoding="utf-8"
    )

    output_files: list[Path] = []
    cells: list[dict[str, Any]] = []

    for length in lengths:
        length_root = _length_data_root(data_root, length)
        for task in tasks:
            examples = load_examples(length_root, task, limit=n)
            # Baseline first, so selection/compression arms can be compared to it.
            ordered_arms = (
                [BASELINE_ARM] + [a for a in arms if a != BASELINE_ARM]
                if BASELINE_ARM in arms
                else list(arms)
            )
            rows_by_arm: dict[str, list[dict[str, Any]]] = {}
            for arm_name in ordered_arms:
                path, rows = run_cell(
                    arm=arm_objects[arm_name],
                    arm_name=arm_name,
                    length=length,
                    task=task,
                    examples=examples,
                    out_root=out_root,
                    client=client,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    budget=budget,
                    operating_point=operating_point,
                    rerun=rerun,
                )
                output_files.append(path)
                rows_by_arm[arm_name] = rows

            baseline_rows = rows_by_arm.get(BASELINE_ARM, [])
            for arm_name in ordered_arms:
                cell = summarize_cell(length, task, arm_name, rows_by_arm[arm_name])
                if arm_name != BASELINE_ARM and baseline_rows:
                    cell["vs_baseline"] = compare_to_baseline(
                        rows_by_arm[arm_name],
                        baseline_rows,
                        seed=seed,
                        n_resamples=bootstrap_resamples,
                    )
                cells.append(cell)

    matrix_json = write_matrix(out_root, config, cells)
    combined_json = write_combined_summary(out_root, cells)

    # Cover every generated artifact, not just the per-item rows. The headline
    # figures live in matrix.json; leaving it out of the manifest meant
    # tools/verify_manifests.py passed on a tampered result file.
    for extra in (
        matrix_json,
        out_root / "matrix.md",
        combined_json,
        out_root / "combined_summary.md",
        out_root / "run_config.json",
        out_root / "trusted_benchmark_manifest.json",
    ):
        if extra.exists() and extra not in output_files:
            output_files.append(extra)

    manifest_path = write_output_manifest(
        out_root / "manifest.sha256", output_files, base_dir=out_root
    )

    return {
        "out_root": str(out_root),
        "run_name": run_name,
        "n_cells": len(cells),
        "n_output_files": len(output_files),
        "matrix": str(out_root / "matrix.json"),
        "combined_summary": str(out_root / "combined_summary.json"),
        "output_manifest": str(manifest_path),
        "trusted_manifest": str(out_root / "trusted_benchmark_manifest.json"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official RULER JSONL through context-selection arms end-to-end."
    )
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="comma-separated arm names: full_context,needlepath,llmlingua2,cpc")
    parser.add_argument("--tasks", default=DEFAULT_TASKS_ARG)
    parser.add_argument("--lengths", default=",".join(DEFAULT_LENGTHS),
                        help="comma-separated length labels (e.g. 8k,16k)")
    parser.add_argument("--n", type=int, default=100, help="examples per (length, task)")
    parser.add_argument("--data-root", type=Path, default=Path("data/ruler"),
                        help="dataset root; files at <data-root>/<length>/<task>/test.jsonl")
    parser.add_argument("--needlepath-url", default=None,
                        help="base URL of the hosted Needlepath endpoint (required for the needlepath arm)")
    parser.add_argument("--operating-point", default=DEFAULT_OPERATING_POINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("runs/ruler"))
    parser.add_argument("--model", default="")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--request-timeout-ms", type=int, default=120000)
    parser.add_argument("--run-name",
                        default=f"ruler_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--rerun", action="store_true", help="ignore existing per-item rows")
    parser.add_argument("--require-clean-worktree", action="store_true",
                        help="abort if the git worktree is dirty")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.require_clean_worktree:
        enforce_clean_worktree()

    arms = _parse_csv(args.arms)
    tasks = _parse_csv(args.tasks)
    lengths = _parse_csv(args.lengths)

    # Estimate spend and abort BEFORE constructing the client or making any call.
    estimate = enforce_budget(
        arms=arms,
        tasks=tasks,
        lengths=lengths,
        n=args.n,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"estimated spend upper bound: ${estimate:.2f}", file=sys.stderr)

    model = resolve_model(args.model or None)
    client = build_gemini_client(timeout_ms=args.request_timeout_ms)

    result = run_suite(
        arms=arms,
        tasks=tasks,
        lengths=lengths,
        n=args.n,
        data_root=args.data_root,
        out_root=args.out,
        client=client,
        model=model,
        max_output_tokens=args.max_output_tokens,
        operating_point=args.operating_point,
        needlepath_url=args.needlepath_url,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        run_name=args.run_name,
        rerun=args.rerun,
        request_timeout_ms=args.request_timeout_ms,
    )
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
