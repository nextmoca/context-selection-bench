"""Shared scoring + paired-statistics for the context-selection benchmark.

This module is the harness's single correctness surface for turning per-item
model outputs into scores, and per-arm results into aggregate comparisons. It
is deliberately pure (numpy / scipy / stdlib only, no engine or arm internals)
so the published numbers reproduce from these formulas alone.

It provides three kinds of thing:

- **Task correctness.** ``parse_final_numeric_answer`` / ``score_exact_match``
  implement the standard GSM8K exact-match convention used by common
  eval harnesses: parse a ``#### <number>`` marker if present (GSM8K's own
  gold-answer format), otherwise fall back to the LAST standalone number in
  the completion, on the assumption a model that omitted the marker still
  stated its final numeric answer at/near the end. This is pure parsing logic,
  unit-testable with hand-written completion strings; it calls no model API.

- **Aggregates over arm results.** ``fallback_rate``, ``aggregate_latency``,
  ``aggregate_token_reduction`` and ``gold_evidence_metric`` read only the
  neutral fields already present on a ``ContextResponse`` (latency, reduction
  ratio, fallback flag, and the optional coverage/answerability summary). They
  aggregate pre-computed fields; they never recompute them.

- **Paired statistics.** ``macro_accuracy``, ``bootstrap_ci_delta`` and
  ``mcnemar_test`` are the load-bearing comparison layer: a per-base-problem
  ("macro") accuracy, a paired bootstrap CI on a micro/EM accuracy delta, and
  McNemar's test for two methods scored on the same items.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, chi2

from csbench.contracts import ContextResponse


_GOLD_MARKER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def parse_final_numeric_answer(model_completion: str) -> Optional[float]:
    """Parse the final numeric answer out of a model completion string.

    Convention (see module docstring): prefer the ``#### <number>`` marker;
    fall back to the last standalone number in the text. Returns ``None`` if no
    number can be found at all (an unparseable/refused completion).
    """
    text = model_completion or ""

    marker_match = _GOLD_MARKER_RE.search(text)
    if marker_match:
        return _to_float(marker_match.group(1))

    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _to_float(numbers[-1])


def _to_float(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def score_exact_match(model_completion: str, gold_numeric: float) -> bool:
    """GSM8K exact-match (EM): True iff the parsed final numeric answer
    equals ``gold_numeric`` (float-tolerant comparison so formatting artifacts
    like a trailing ``.0`` don't spuriously fail an otherwise-correct answer).
    """
    parsed = parse_final_numeric_answer(model_completion)
    if parsed is None:
        return False
    return math.isclose(parsed, gold_numeric, rel_tol=1e-9, abs_tol=1e-9)


@dataclass(frozen=True)
class GoldEvidenceMetric:
    """Per-item view of the ONE pre-registered gold-evidence metric.

    ``applicable=False`` for arms that never produce a coverage/answerability
    summary (pure compression baselines): this axis is reported as explicitly
    not-applicable for them, never forced to a number.
    """

    applicable: bool
    coverage_score: Optional[float] = None
    selection_safe: Optional[bool] = None
    fallback_required: Optional[bool] = None
    evidence_shape: Optional[str] = None


def gold_evidence_metric(result: ContextResponse) -> GoldEvidenceMetric:
    """Surface the pre-registered gold-evidence view for one arm result.

    Reads the neutral coverage/answerability summary already computed by the
    arm (``ContextResponse.safety``). Arms that never produce such a summary
    (pure compression / full-context baselines) leave it ``None``, and this
    axis is reported as ``applicable=False`` for them, never forced to a
    number.
    """
    safety = result.safety
    if safety is None:
        return GoldEvidenceMetric(applicable=False)
    return GoldEvidenceMetric(
        applicable=True,
        coverage_score=safety.coverage_score,
        selection_safe=safety.selection_safe,
        fallback_required=safety.fallback_required,
        evidence_shape=str(safety.evidence_shape),
    )


@dataclass(frozen=True)
class LatencyAggregate:
    n: int
    total_latency_ms: float
    mean_latency_ms: float


def aggregate_latency(results: Sequence[ContextResponse]) -> LatencyAggregate:
    """Aggregate the wall-clock latency already measured on each result
    (``ContextResponse.engine_latency_ms``): does not recompute latency."""
    n = len(results)
    if n == 0:
        return LatencyAggregate(
            n=0,
            total_latency_ms=0.0,
            mean_latency_ms=0.0,
        )
    total_latency_ms = sum(r.engine_latency_ms for r in results)
    return LatencyAggregate(
        n=n,
        total_latency_ms=total_latency_ms,
        mean_latency_ms=total_latency_ms / n,
    )


def aggregate_token_reduction(results: Sequence[ContextResponse]) -> Optional[float]:
    """Mean full-prompt-basis reduction ratio across results, or ``None`` if
    there are none.

    FULL-PROMPT-BASIS: ``ContextResponse.reduction_ratio`` is computed over the
    whole prompt (including any fixed, always-present scaffolding in both its
    numerator and denominator): this is the real economic token-savings number
    a user would actually see.
    """
    if not results:
        return None
    return sum(r.reduction_ratio for r in results) / len(results)


def fallback_rate(results: Sequence[ContextResponse]) -> Optional[float]:
    """Fraction of results with ``fallback_used=True``, or ``None`` for empty
    input.

    Fail-open/fallback is specific to selection methods; pure compression /
    full-context baselines never fall back by construction, so this metric is
    meaningful only when every result comes from a selection arm. Callers are
    responsible for passing an arm-homogeneous set of results.
    """
    if not results:
        return None
    return sum(1 for r in results if r.fallback_used) / len(results)


# ---------------------------------------------------------------------------
# macro_accuracy (per-base-problem accuracy)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MacroAccuracyResult:
    """Result of a macro (per-base-problem) accuracy computation.

    See ``macro_accuracy``'s docstring for the CRITICAL caveat about ``n``: the
    denominator here is ``n_base_problems`` (the number of DISTINCT groups),
    NOT the item-level count that went in. Never headline/emphasize a macro
    delta over the micro (per-example) accuracy delta in a report: macro's
    true N is small and its CI is correspondingly wide.
    """

    accuracy: float
    n_base_problems: int
    n_items: int
    failed_base_problem_ids: List[int]


def macro_accuracy(item_correctness: Sequence[Tuple[int, bool]]) -> MacroAccuracyResult:
    """Per-BASE-PROBLEM ("macro") accuracy: a base problem counts as
    "correct" only if ALL of its sampled variant items are individually
    correct (EM=True).

    Motivating use case (GSM-IC-style distractor sweeps): a stratified sample
    draws several variant items per base problem. Within one method x budget x
    condition cell, this function groups the scored items by
    ``base_problem_id`` and asks: for how many of the base problems did the
    model get EVERY sampled variant right? That fraction is "macro accuracy."

    This function is otherwise fully generic over any ``(group_id,
    is_correct)`` pairs (it does not import or depend on anything
    dataset-specific) but it is motivated by, and should be read against, a
    small-number-of-base-problems structure.

    *** CRITICAL CAVEAT ***: macro accuracy's true sample size is the number of
    DISTINCT base problems (``n_base_problems`` on the returned result), which
    is typically SMALL, NOT the item-level N that was passed in. A macro
    accuracy computed over only a few dozen independent units has a
    correspondingly WIDE confidence interval. Macro deltas between methods
    should NEVER be headlined or emphasized over the micro (per-example)
    accuracy delta, which is the load-bearing metric for this benchmark:
    macro accuracy is a secondary, coarser-grained sanity check ("does the
    method get whole problems fully right"), not the primary comparison.

    Args:
        item_correctness: sequence of ``(base_problem_id, is_correct)`` tuples,
            one tuple per scored item. Multiple tuples will typically share
            the same ``base_problem_id`` (one per sampled variant of that base
            problem in this method x budget x condition cell).

    Returns:
        A ``MacroAccuracyResult`` with the macro accuracy, the number of
        distinct base problems (the true denominator), the total item count
        that went in (for context only, NOT the macro denominator), and the
        list of base-problem ids that were NOT all-correct (useful detail
        for a report).

    Raises:
        ValueError: if ``item_correctness`` is empty.
    """
    if not item_correctness:
        raise ValueError("macro_accuracy requires at least one (base_problem_id, is_correct) tuple")

    groups: Dict[int, List[bool]] = {}
    group_order: List[int] = []
    for base_problem_id, is_correct in item_correctness:
        if base_problem_id not in groups:
            groups[base_problem_id] = []
            group_order.append(base_problem_id)
        groups[base_problem_id].append(is_correct)

    failed_base_problem_ids = [
        base_problem_id
        for base_problem_id in group_order
        if not all(groups[base_problem_id])
    ]
    n_base_problems = len(group_order)
    n_all_correct = n_base_problems - len(failed_base_problem_ids)

    return MacroAccuracyResult(
        accuracy=n_all_correct / n_base_problems,
        n_base_problems=n_base_problems,
        n_items=len(item_correctness),
        failed_base_problem_ids=failed_base_problem_ids,
    )


# ---------------------------------------------------------------------------
# bootstrap_ci_delta: paired bootstrap CI on a micro/EM accuracy delta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapCIResult:
    """Result of a paired-bootstrap confidence interval on an accuracy
    delta between two methods (``mean(correctness_a) - mean(correctness_b)``)."""

    observed_delta: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_items: int


def bootstrap_ci_delta(
    correctness_a: Sequence[bool],
    correctness_b: Sequence[bool],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResult:
    """Bootstrap confidence interval on a micro/EM accuracy DELTA between
    two methods (e.g. a selection arm vs. the full-context baseline at a
    matched budget/condition): the load-bearing comparison this benchmark
    cares about.

    PAIRED resampling: ``correctness_a`` and ``correctness_b`` are assumed to be
    genuinely PAIRED (the same items, in the same order) which is the
    normal case here, since every method scores the exact same item set at a
    given budget/condition. On each of ``n_resamples`` iterations, a single
    set of indices (size = original N) is resampled WITH REPLACEMENT and
    used to index into BOTH arrays together (never resampled
    independently), preserving the pairing. This is the statistically
    correct approach for a paired accuracy delta (an independent-resample
    bootstrap would overstate the delta's variance by ignoring the
    correlation between ``correctness_a[i]`` and ``correctness_b[i]`` for the
    same item ``i``).

    Callers are responsible for passing genuinely paired/aligned sequences
    (same items, same order in both arrays); this function only validates
    that the lengths match and cannot detect a silent misalignment.

    Reproducibility: resampling uses ``np.random.default_rng(seed)``, so the
    same ``(correctness_a, correctness_b, n_resamples, seed)`` always
    produces the same CI.

    Args:
        correctness_a: per-item correctness (bool) for method A.
        correctness_b: per-item correctness (bool) for method B, paired
            index-for-index with ``correctness_a``.
        n_resamples: number of bootstrap resamples.
        confidence: confidence level for the returned CI (e.g. 0.95 for a
            95% CI, using the 2.5th/97.5th percentiles of the resampled
            delta distribution).
        seed: seed for the reproducible RNG.

    Returns:
        A ``BootstrapCIResult`` with the observed (non-resampled) delta and
        the percentile CI from the resampled distribution.

    Raises:
        ValueError: if ``correctness_a`` and ``correctness_b`` have different
            lengths (paired resampling requires equal length and matched
            ordering).
    """
    if len(correctness_a) != len(correctness_b):
        raise ValueError(
            f"correctness_a (len={len(correctness_a)}) and correctness_b "
            f"(len={len(correctness_b)}) must be the same length for paired "
            "bootstrap resampling"
        )

    n_items = len(correctness_a)
    arr_a = np.asarray(correctness_a, dtype=float)
    arr_b = np.asarray(correctness_b, dtype=float)

    observed_delta = float(arr_a.mean() - arr_b.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample_idx = rng.integers(0, n_items, size=n_items)
        deltas[i] = arr_a[resample_idx].mean() - arr_b[resample_idx].mean()

    alpha = 1.0 - confidence
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BootstrapCIResult(
        observed_delta=observed_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_resamples=n_resamples,
        n_items=n_items,
    )


# ---------------------------------------------------------------------------
# mcnemar_test: paired McNemar's test for two methods on the same items
# ---------------------------------------------------------------------------

# Threshold on the discordant-pair count (b + c) above which the
# continuity-corrected chi-square approximation is used; below it, the exact
# binomial test is used instead. 25 is a common convention for McNemar's test
# (the chi-square approximation is unreliable for small discordant counts).
_MCNEMAR_CHI2_THRESHOLD = 25


@dataclass(frozen=True)
class McNemarResult:
    """Result of McNemar's test for paired accuracy between two methods on
    the same items."""

    b: int
    c: int
    statistic: Optional[float]
    p_value: float
    method: str
    n_items: int


def mcnemar_test(correctness_a: Sequence[bool], correctness_b: Sequence[bool]) -> McNemarResult:
    """McNemar's test for paired accuracy: the standard test for "did
    method A get item i right/wrong differently from method B, on the SAME
    items", exactly this benchmark's matched-budget, same-item, two-method
    comparison (e.g. a selection arm vs. the full-context baseline).

    Builds the standard 2x2 discordant-pair contingency:
      - ``b`` = count where A is correct AND B is incorrect
      - ``c`` = count where A is incorrect AND B is correct
    Concordant pairs (both correct, or both incorrect) do not matter for
    McNemar's test and are not counted.

    Uses the continuity-corrected chi-square statistic
    ``(|b - c| - 1)^2 / (b + c)`` (chi-square with 1 degree of freedom, via
    ``scipy.stats.chi2.sf``) when ``b + c >= 25`` (see ``_MCNEMAR_CHI2_THRESHOLD``:
    25 is a common convention for when the chi-square approximation to
    the binomial is reliable); falls back to the EXACT binomial test
    (``scipy.stats.binomtest(min(b, c), b + c, 0.5)``) for smaller ``b + c``,
    including the ``b + c == 0`` degenerate case (no discordant pairs at all:
    handled explicitly by returning ``p_value=1.0`` as a documented sentinel,
    with ``method="exact_binomial"`` and ``statistic=None``, rather than
    dividing by zero).

    Args:
        correctness_a: per-item correctness (bool) for method A.
        correctness_b: per-item correctness (bool) for method B, paired
            index-for-index with ``correctness_a`` (same items, same order).

    Returns:
        A ``McNemarResult`` with the discordant-pair counts, the chi-square
        statistic (``None`` if the exact test was used), the p-value, and
        which ``method`` was used (``"chi2_continuity_corrected"`` or
        ``"exact_binomial"``) so a report can state which test applies.

    Raises:
        ValueError: if ``correctness_a`` and ``correctness_b`` have different
            lengths.
    """
    if len(correctness_a) != len(correctness_b):
        raise ValueError(
            f"correctness_a (len={len(correctness_a)}) and correctness_b "
            f"(len={len(correctness_b)}) must be the same length for a "
            "paired McNemar test"
        )

    n_items = len(correctness_a)
    b = sum(1 for a_val, b_val in zip(correctness_a, correctness_b) if a_val and not b_val)
    c = sum(1 for a_val, b_val in zip(correctness_a, correctness_b) if not a_val and b_val)

    if b + c == 0:
        return McNemarResult(
            b=b, c=c, statistic=None, p_value=1.0, method="exact_binomial", n_items=n_items
        )

    if b + c >= _MCNEMAR_CHI2_THRESHOLD:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(chi2.sf(statistic, df=1))
        return McNemarResult(
            b=b,
            c=c,
            statistic=float(statistic),
            p_value=p_value,
            method="chi2_continuity_corrected",
            n_items=n_items,
        )

    exact = binomtest(min(b, c), b + c, 0.5)
    return McNemarResult(
        b=b, c=c, statistic=None, p_value=float(exact.pvalue), method="exact_binomial", n_items=n_items
    )
