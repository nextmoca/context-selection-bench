"""Paired statistics (McNemar + bootstrap CI) for the real RULER
matched-protocol benchmark (Task 4 of the benchmark design).

Spec (verbatim, Task 4): "New module ... that reads the persisted per-item
`rows.json` (already written per task by `run_task`) for both context
lengths, and for each (task, length) pair computes: Needlepath vs
full_context: `mcnemar_test` on paired correctness (row-aligned by
`index`/`position`) ... Needlepath vs llmlingua2: same. `bootstrap_ci_delta`
on the score delta for both comparisons."

This module does NOT reimplement `mcnemar_test`/`bootstrap_ci_delta` --
both are imported directly from `csbench.stats`, which already unit-tests their
internal math. This module is only responsible for: (1) loading the real,
already-persisted `rows.json` files (Task 2's `run_task` output shape), (2)
validating row alignment between the two methods being compared, (3)
deciding + documenting a binarization rule so McNemar (which needs paired
BOOLEAN correctness) can be run on RULER's float-valued scores, and (4)
formatting the per-(task, length) results as JSON + a markdown table.

---------------------------------------------------------------------------
On-disk shape this module consumes (confirmed against `harness.py`)
---------------------------------------------------------------------------

`harness.run_task` (see its `task_dir = output_dir / task` /
`(task_dir / "rows.json").write_text(...)` lines) writes one JSON array per
task to `<output_dir>/<task>/rows.json`. Each element is a per-item dict
(built by `_run_case`) that includes -- among many other diagnostic fields
this module ignores -- all of:

    "index"              -- the example's stable RULER index (int)
    "position"           -- 1-based position in this task's example loop
    "baseline_score"      -- float in [0, 1], full-context baseline's score
    "needlepath_score"    -- float in [0, 1], Needlepath's score
    "llmlingua2_score"     -- float in [0, 1], LLMLingua-2's score

All three method scores come from the SAME row (one call to `_run_case`
computes `baseline`, `needlepath`, and `llmlingua2` completions for the same
example in the same loop iteration) -- so "row alignment" is trivially
guaranteed by construction for a single, uncorrupted `rows.json`. This
module still validates it explicitly (rather than assuming it) so a
corrupted/hand-edited/partially-written file (e.g. a duplicate `index`, or a
method's score missing/None for some row due to a partial/errored run)
fails loudly instead of silently zipping mismatched items together.

`run_ruler_context`/`run_ruler_suite` write these per-task directories under
`<output_root>/<run_name>_<context_label>/<task>/rows.json`, where
`context_label` is `"8k"` or `"16k"` (see `harness.run_ruler_suite`:
`context_output = output_root / f"{run_name}_{label}"`, and
`DEFAULT_DATA_ROOTS`/`RULER_LENGTH_LABELS` in `cost.py`
confirm the label vocabulary is exactly `{"8k", "16k"}`). This module's
`rows_path_for` builds exactly that path so it is ready to point at the
real Task 5 output once it exists -- no real `rows.json` files exist yet as
of Task 4 (Task 5, the actual spend, has not run); this module is developed
and tested entirely against synthetic, hand-built `rows.json`-shaped
fixtures.

---------------------------------------------------------------------------
Binarization decision for McNemar (RULER scores are floats, not booleans)
---------------------------------------------------------------------------

`score_ruler_answer` (`harness.py`) computes two different formulas
depending on a task's `match_type` (`match_type_for_task`):

  - `match_type == "part"` (the `qa_*` tasks): `max(1.0 if ref in prediction
    else 0.0 for ref in references)` -- this is ALREADY always exactly 0.0
    or 1.0 (a max over a set of 0/1 values), so binarization is a no-op for
    these tasks.
  - `match_type == "all"` (every other task, including `niah_multivalue`/
    `niah_multiquery`/`vt`, which have multiple gold references): `sum(1.0
    if ref in prediction else 0.0 for ref in references) / len(references)`
    -- this IS an average, so it CAN be fractional (e.g. 2/3 references
    found) whenever a multi-reference item is partially matched.

Decision: **binarize with `score >= 1.0`** (`MCNEMAR_CORRECTNESS_THRESHOLD`
below) -- i.e. "correct" for McNemar means ALL gold references were found,
matching `match_type == "all"`'s own semantics (the task is defined to
require every reference, so partial credit there is not "task success" in
the sense McNemar's discordant-pair test cares about: "did this method get
the item right or wrong"). This is a strict/conservative choice: a
99%-of-references item is treated as "wrong," same as a 0%-of-references
item. We use it rather than a `> 0` ("any credit") threshold because (a) it
requires no invented cutoff -- `>= 1.0` is just "fully correct," the same
bar the task's own scoring formula is built around, and (b) `bootstrap_ci_delta`
(run on the RAW float scores, not binarized -- see below) already reports
the softer, partial-credit-sensitive picture of the same comparison, so the
two metrics are complementary rather than redundant: McNemar/binarized
answers "did the method flip strict correctness on individual items,"
bootstrap/raw answers "what is the mean score delta including partial
credit." For `match_type == "part"` tasks this threshold is inert (scores
are already binary).

`bootstrap_ci_delta` is run on the RAW (un-binarized) float scores -- the
spec explicitly says "bootstrap_ci_delta on the score delta," and
`bootstrap_ci_delta`'s own docstring documents it as generic over any
paired numeric sequences (it only calls `np.asarray(..., dtype=float)` and
takes means/deltas -- nothing GSM8K/boolean-specific), so no binarization
is needed or wanted for that metric.

---------------------------------------------------------------------------
CPC arm extension (Task 7 of the benchmark design)
---------------------------------------------------------------------------

Optional, opt-in via `--cpc-run-name`/`compute_ruler_stats(..., cpc_run_name=...)`.
When given, a cpc-only pass's `rows.json` (same on-disk layout, produced by
the CPC arm's `complete_case` -- `cpc_score` plus `cpc_*` diagnostics) is
joined onto the main run's rows BY POSITION (`join_cpc_rows`), with a hard
error on any position-set mismatch or `index` identity mismatch between the
two sides -- unpaired/misidentified data must never silently pass. Three
additional comparisons are then computed (`CPC_COMPARISONS`): the headline
`cpc_vs_needlepath`, plus `cpc_vs_full_context` and `cpc_vs_llmlingua2`.
Sign convention (documented on `CPC_COMPARISONS`): delta = cpc's score minus
the other named method's score, consistent with `COMPARISONS`' own "method A
minus method B" convention. `compute_cpc_aggregates` additionally reports,
per comparison and length, the mean bootstrap delta across tasks in two
scopes -- `"all_tasks"` and the headline `"headline_excluding_niah_single_3"`
(dropping the known `niah_single_3` dead-zero artifact, see
the published RULER results' "Known limitations") -- side by side, so
the exclusion is always visible rather than silently applied; per-(task,
length) rows are unaffected and always include every task. Without
`cpc_run_name` (the default), this module's behavior is unchanged.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from csbench.stats import BootstrapCIResult, McNemarResult, bootstrap_ci_delta, mcnemar_test

# See module docstring's "Binarization decision" section.
MCNEMAR_CORRECTNESS_THRESHOLD = 1.0

# Matches `cost.py`'s `RULER_LENGTH_LABELS` / the RULER harness's
# `run_ruler_suite` context labels.
DEFAULT_LENGTH_LABELS: Sequence[str] = ("8k", "16k")

DEFAULT_ROWS_FILENAME = "rows.json"

# (comparison_name, method_a_field, method_b_field) -- method A is always
# Needlepath, per the spec's two named comparisons.
COMPARISONS: Sequence[Tuple[str, str, str]] = (
    ("needlepath_vs_full_context", "needlepath_score", "baseline_score"),
    ("needlepath_vs_llmlingua2", "needlepath_score", "llmlingua2_score"),
)

# CPC arm (Task 7 of the benchmark design): three additional
# comparisons against the cpc-only pass's `cpc_score` field, joined onto the
# main 3-arm run's rows by `position` (see `join_cpc_rows` below). Method A
# is always cpc, per the spec's "delta = first_named_method - second (i.e.
# cpc - needlepath)" sign convention -- so for every comparison here,
# `ComparisonStats.bootstrap.observed_delta` is `mean(cpc_score) -
# mean(other_method_score)`, and `ComparisonStats.mcnemar.b`/`.c` count
# discordant pairs with cpc as the "row" method
# (the scorers module's `mcnemar_test(correctness_a, correctness_b)` convention).
# `cpc_vs_needlepath` is the headline comparison (cpc vs. the project's own
# selective-state method); the other two situate cpc relative to the
# existing full_context/llmlingua2 arms.
CPC_COMPARISONS: Sequence[Tuple[str, str, str]] = (
    ("cpc_vs_needlepath", "cpc_score", "needlepath_score"),
    ("cpc_vs_full_context", "cpc_score", "baseline_score"),
    ("cpc_vs_llmlingua2", "cpc_score", "llmlingua2_score"),
)

# Known dead-zero artifact (see the published RULER results' "Known
# limitations": all three original methods score 0.0% on this task at both
# context lengths due to a task-construction/UUID-truncation issue, not a
# method-level finding). Per-(task, length) rows always include it; only the
# cpc "headline" aggregate (see `compute_cpc_aggregates`) excludes it.
NIAH_SINGLE_3_TASK = "niah_single_3"


# ---------------------------------------------------------------------------
# Row loading + alignment
# ---------------------------------------------------------------------------


def rows_path_for(output_root: Path, run_name: str, length_label: str, task: str) -> Path:
    """The real on-disk path `run_ruler_suite`/`run_ruler_context`/`run_task`
    write a task's rows to: `<output_root>/<run_name>_<length_label>/<task>/rows.json`."""
    return output_root / f"{run_name}_{length_label}" / task / DEFAULT_ROWS_FILENAME


def load_rows(rows_path: Path) -> List[Dict[str, Any]]:
    """Load a `rows.json` file (a JSON array of per-item dicts, per
    `run_task`'s on-disk shape)."""
    if not rows_path.exists():
        raise FileNotFoundError(f"rows.json not found at {rows_path}")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{rows_path}: expected a JSON array of row dicts, got {type(rows).__name__}")
    return rows


def _extract_method_scores(rows: Sequence[Dict[str, Any]], field: str) -> Dict[int, float]:
    """Build a `{position: score}` map for one method's score `field`, from a
    shared `rows` list.

    NOTE (found against the real Task 5 output, not the synthetic fixtures
    this module was originally developed against): RULER's own example
    `index` field is NOT guaranteed unique within a single task's 100-row
    `rows.json` -- the real run has many tasks (e.g. `niah_multikey_3`,
    `niah_multivalue`, several `niah_single_*`) with genuine duplicate
    `index` values across different rows (verified: still exactly 100 rows,
    100 real API calls, `position` 1..100 always unique). `index` appears to
    be RULER's own generator's sample identifier, which is not guaranteed
    distinct within a task's sampled set. Row alignment therefore uses the
    harness's own `position` field (1-based loop position, always unique per
    task per `run_task`) instead of `index`.

    A row whose `field` value is missing/`None` is treated as "this method
    has no usable score for this item" and is simply excluded from the
    returned map (rather than raising) -- so that a genuine cross-method
    misalignment (one method's score present, the other's absent, for the
    same item) is detectable by comparing the two maps' key sets in
    `paired_scores_by_index` below, instead of being masked here.

    Raises:
        ValueError: if any row is missing the `position` field, or if two
            rows report the same `position` for this `field` (a genuinely
            corrupted/duplicated `rows.json` -- fail loudly rather than
            silently overwrite).
    """
    scores: Dict[int, float] = {}
    for row in rows:
        if "position" not in row or row["position"] is None:
            raise ValueError(f"row missing required 'position' field: {row!r}")
        position = int(row["position"])
        value = row.get(field)
        if value is None:
            continue
        if position in scores:
            raise ValueError(
                f"duplicate position {position} found while extracting '{field}' scores "
                "-- rows.json appears corrupted (expected one row per position)"
            )
        scores[position] = float(value)
    return scores


def paired_scores_by_index(
    rows: Sequence[Dict[str, Any]], field_a: str, field_b: str
) -> Tuple[List[int], List[float], List[float]]:
    """Extract two methods' scores from the same `rows` list, row-aligned by
    `position` (see `_extract_method_scores`'s docstring for why `position`
    is used instead of RULER's own `index` field), validating that both
    methods have scores for exactly the same set of items.

    Returns `(positions, scores_a, scores_b)`, all three lists sorted
    ascending by `position` and mutually aligned position-for-position.
    (Function/parameter names keep the `index`/`indices` wording for
    backward compatibility with callers; the values are `position`s.)

    Raises:
        ValueError: if the set of positions with a usable score for
            `field_a` does not exactly match the set for `field_b` (row
            misalignment -- see module docstring), or if there are zero
            common positions.
    """
    scores_a = _extract_method_scores(rows, field_a)
    scores_b = _extract_method_scores(rows, field_b)

    indices_a = set(scores_a)
    indices_b = set(scores_b)
    if indices_a != indices_b:
        only_a = sorted(indices_a - indices_b)
        only_b = sorted(indices_b - indices_a)
        raise ValueError(
            f"row misalignment between '{field_a}' and '{field_b}': "
            f"positions only in '{field_a}': {only_a}; positions only in '{field_b}': {only_b}"
        )

    if not indices_a:
        raise ValueError(f"no rows with usable scores for both '{field_a}' and '{field_b}'")

    common_indices = sorted(indices_a)
    values_a = [scores_a[index] for index in common_indices]
    values_b = [scores_b[index] for index in common_indices]
    return common_indices, values_a, values_b


def binarize_correctness(
    scores: Sequence[float], *, threshold: float = MCNEMAR_CORRECTNESS_THRESHOLD
) -> List[bool]:
    """Binarize float RULER scores into booleans for McNemar's test. See the
    module docstring's "Binarization decision" section for why `>= 1.0` is
    the default threshold."""
    return [score >= threshold for score in scores]


# ---------------------------------------------------------------------------
# CPC arm: joining a cpc-only rows.json onto the main 3-arm rows.json
# ---------------------------------------------------------------------------


def _index_rows_by_position(rows: Sequence[Dict[str, Any]], *, source: str) -> Dict[int, Dict[str, Any]]:
    """Build a `{position: row}` map for a whole `rows.json`, failing loudly
    on a missing `position` field or a duplicate `position` (a corrupted
    rows.json) -- used by `join_cpc_rows` to index both sides before
    joining."""
    indexed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if "position" not in row or row["position"] is None:
            raise ValueError(f"{source}: row missing required 'position' field: {row!r}")
        position = int(row["position"])
        if position in indexed:
            raise ValueError(
                f"{source}: duplicate position {position} -- rows.json appears corrupted "
                "(expected one row per position)"
            )
        indexed[position] = row
    return indexed


def join_cpc_rows(
    main_rows: Sequence[Dict[str, Any]], cpc_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Join a cpc-only pass's rows (see the CPC arm's `complete_case`
    return shape) onto the main 3-arm run's rows, for one (task, length)
    cell -- BY POSITION, matching the rest of this module's alignment
    convention (`_extract_method_scores`'s docstring explains why `position`
    is used instead of RULER's own non-unique `index`).

    Two hard-error checks, per the cpc run's binding constraint that any
    cpc result rows must pair 1:1 by `position` with the existing rows
    (the CPC run's global constraints):

    1. **Position-set equality**: `main_rows` and `cpc_rows` must cover
       EXACTLY the same set of positions. Any position present on only one
       side (a short/partial cpc pass, a shifted/off-by-one position, etc.)
       is a hard error listing the specific missing/extra positions --
       unpaired data must never silently pass (e.g. by falling back to an
       inner join over the intersection).
    2. **Identity cross-check**: for every position present on both sides,
       the two rows' `index` fields must also agree. A same-`position`-
       different-`index` pair means the cpc pass ran over a DIFFERENT
       underlying item than the main run did for that position -- an
       identity violation, not a mere pairing gap -- so it is reported as
       a distinct hard error rather than folded into the position check.

    Returns one merged row per position (sorted ascending by position): the
    main row's own fields, plus every `cpc_*`-prefixed field from the
    matching cpc row (so `cpc_score` sits alongside the existing
    `needlepath_score`/`baseline_score`/`llmlingua2_score` fields, ready for
    `compute_comparison`/`compute_task_length_stats` unchanged). Only
    `cpc_*`-prefixed fields are copied over (rather than blindly merging the
    whole cpc row) so the cpc row's own redundant copies of shared fields
    like `task`/`length`/`expected_answer`/`match_type` never silently
    shadow the main row's already-validated values.
    """
    main_by_position = _index_rows_by_position(main_rows, source="main rows.json")
    cpc_by_position = _index_rows_by_position(cpc_rows, source="cpc rows.json")

    main_positions = set(main_by_position)
    cpc_positions = set(cpc_by_position)
    if main_positions != cpc_positions:
        missing_in_cpc = sorted(main_positions - cpc_positions)
        missing_in_main = sorted(cpc_positions - main_positions)
        raise ValueError(
            "cpc join failed: position sets do not match between main run and cpc run -- "
            f"positions missing from cpc rows.json: {missing_in_cpc}; "
            f"positions present in cpc rows.json but missing from main rows.json: {missing_in_main}"
        )

    joined: List[Dict[str, Any]] = []
    for position in sorted(main_positions):
        main_row = main_by_position[position]
        cpc_row = cpc_by_position[position]
        main_index = main_row.get("index")
        cpc_index = cpc_row.get("index")
        if main_index != cpc_index:
            raise ValueError(
                f"cpc join failed: identity violation at position {position} -- "
                f"main row 'index' is {main_index!r}, cpc row 'index' is {cpc_index!r} "
                "(the same position must refer to the same underlying item on both sides)"
            )
        merged = dict(main_row)
        for key, value in cpc_row.items():
            if key.startswith("cpc_"):
                merged[key] = value
        joined.append(merged)
    return joined


# ---------------------------------------------------------------------------
# Per-comparison / per-(task, length) stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonStats:
    """McNemar (on binarized correctness) + bootstrap CI (on the raw float
    score delta) for one Needlepath-vs-other-method comparison, on one
    (task, length) cell."""

    comparison: str
    method_a_field: str
    method_b_field: str
    n_items: int
    mcnemar_threshold: float
    mcnemar: McNemarResult
    bootstrap: BootstrapCIResult


def compute_comparison(
    rows: Sequence[Dict[str, Any]],
    comparison: str,
    method_a_field: str,
    method_b_field: str,
    *,
    mcnemar_threshold: float = MCNEMAR_CORRECTNESS_THRESHOLD,
) -> ComparisonStats:
    indices, scores_a, scores_b = paired_scores_by_index(rows, method_a_field, method_b_field)
    correctness_a = binarize_correctness(scores_a, threshold=mcnemar_threshold)
    correctness_b = binarize_correctness(scores_b, threshold=mcnemar_threshold)

    mcnemar = mcnemar_test(correctness_a, correctness_b)
    bootstrap = bootstrap_ci_delta(scores_a, scores_b)

    return ComparisonStats(
        comparison=comparison,
        method_a_field=method_a_field,
        method_b_field=method_b_field,
        n_items=len(indices),
        mcnemar_threshold=mcnemar_threshold,
        mcnemar=mcnemar,
        bootstrap=bootstrap,
    )


@dataclass(frozen=True)
class TaskLengthStats:
    """Both named comparisons (needlepath vs full_context, needlepath vs
    llmlingua2) for one (task, length) cell."""

    task: str
    length_label: str
    n_rows: int
    comparisons: Dict[str, ComparisonStats]


def compute_task_length_stats(
    rows: Sequence[Dict[str, Any]],
    task: str,
    length_label: str,
    *,
    comparisons: Sequence[Tuple[str, str, str]] = COMPARISONS,
    mcnemar_threshold: float = MCNEMAR_CORRECTNESS_THRESHOLD,
) -> TaskLengthStats:
    computed = {
        name: compute_comparison(rows, name, field_a, field_b, mcnemar_threshold=mcnemar_threshold)
        for name, field_a, field_b in comparisons
    }
    return TaskLengthStats(
        task=task,
        length_label=length_label,
        n_rows=len(rows),
        comparisons=computed,
    )


# ---------------------------------------------------------------------------
# Whole-run computation (real rows.json files) + serialization
# ---------------------------------------------------------------------------


def compute_ruler_stats(
    output_root: Path,
    run_name: str,
    tasks: Sequence[str],
    *,
    length_labels: Sequence[str] = DEFAULT_LENGTH_LABELS,
    comparisons: Sequence[Tuple[str, str, str]] = COMPARISONS,
    mcnemar_threshold: float = MCNEMAR_CORRECTNESS_THRESHOLD,
    cpc_run_name: str | None = None,
    cpc_comparisons: Sequence[Tuple[str, str, str]] = CPC_COMPARISONS,
) -> List[TaskLengthStats]:
    """Load the real, persisted `rows.json` for every (task, length) cell
    under `<output_root>/<run_name>_<length_label>/<task>/rows.json` and
    compute both named comparisons for each cell.

    If `cpc_run_name` is given, also load
    `<output_root>/<cpc_run_name>_<length_label>/<task>/rows.json`, join it
    onto the main run's rows by position (`join_cpc_rows` -- hard errors on
    any pairing/identity mismatch), and additionally compute
    `cpc_comparisons` (default `CPC_COMPARISONS`) for that cell. Without
    `cpc_run_name` (the default, `None`), behavior is unchanged from before
    the cpc extension -- no cpc file is read, no cpc comparisons are added.
    """
    cells: List[TaskLengthStats] = []
    for length_label in length_labels:
        for task in tasks:
            rows_path = rows_path_for(output_root, run_name, length_label, task)
            rows = load_rows(rows_path)
            active_comparisons = list(comparisons)
            if cpc_run_name is not None:
                cpc_rows_path = rows_path_for(output_root, cpc_run_name, length_label, task)
                cpc_rows = load_rows(cpc_rows_path)
                rows = join_cpc_rows(rows, cpc_rows)
                active_comparisons = active_comparisons + list(cpc_comparisons)
            cells.append(
                compute_task_length_stats(
                    rows,
                    task,
                    length_label,
                    comparisons=active_comparisons,
                    mcnemar_threshold=mcnemar_threshold,
                )
            )
    return cells


@dataclass(frozen=True)
class CpcAggregate:
    """One aggregate summary line: the mean bootstrap `observed_delta` for
    one cpc comparison, one context length, over a set of tasks (either all
    tasks, or all tasks excluding the flagged `niah_single_3` dead-zone
    artifact -- see `compute_cpc_aggregates`)."""

    comparison: str
    length_label: str
    scope: str
    n_tasks: int
    tasks: List[str]
    mean_observed_delta: float


def compute_cpc_aggregates(
    cells: Sequence[TaskLengthStats],
    *,
    cpc_comparisons: Sequence[Tuple[str, str, str]] = CPC_COMPARISONS,
    excluded_task: str = NIAH_SINGLE_3_TASK,
) -> List[CpcAggregate]:
    """Aggregate each cpc comparison's bootstrap `observed_delta` across
    tasks, grouped by (comparison, length_label), reported in TWO scopes:

    - `"all_tasks"`: mean over every (task, length) cell that has this
      comparison.
    - `"headline_excluding_niah_single_3"`: the same mean, but with
      `excluded_task` (`niah_single_3` by default -- the known dead-zero
      artifact documented in the published RULER results' "Known
      limitations") dropped. This is the HEADLINE number.

    Both scopes are always returned side by side (never just one) so the
    exclusion is visible/auditable rather than silently applied. Per-(task,
    length) rows elsewhere in the output are unaffected by this function --
    they always include every task, `niah_single_3` included; only these
    aggregate lines ever drop it.
    """
    comparison_names = [name for name, _, _ in cpc_comparisons]
    length_labels = sorted({cell.length_label for cell in cells})

    aggregates: List[CpcAggregate] = []
    for comparison_name in comparison_names:
        for length_label in length_labels:
            relevant_cells = [
                cell
                for cell in cells
                if cell.length_label == length_label and comparison_name in cell.comparisons
            ]
            scopes = (
                ("all_tasks", relevant_cells),
                (
                    "headline_excluding_niah_single_3",
                    [cell for cell in relevant_cells if cell.task != excluded_task],
                ),
            )
            for scope, cells_for_scope in scopes:
                if not cells_for_scope:
                    continue
                deltas = [
                    cell.comparisons[comparison_name].bootstrap.observed_delta
                    for cell in cells_for_scope
                ]
                aggregates.append(
                    CpcAggregate(
                        comparison=comparison_name,
                        length_label=length_label,
                        scope=scope,
                        n_tasks=len(cells_for_scope),
                        tasks=sorted(cell.task for cell in cells_for_scope),
                        mean_observed_delta=sum(deltas) / len(deltas),
                    )
                )
    return aggregates


def _comparison_stats_to_dict(stats: ComparisonStats) -> Dict[str, Any]:
    payload = asdict(stats)
    return payload


def cells_to_json_payload(
    cells: Sequence[TaskLengthStats],
    *,
    run_name: str,
    cpc_aggregates: Sequence[CpcAggregate] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "mcnemar_correctness_threshold": MCNEMAR_CORRECTNESS_THRESHOLD,
        "note": (
            "McNemar is computed on binarized correctness (score >= "
            f"{MCNEMAR_CORRECTNESS_THRESHOLD}); bootstrap CI is computed on the "
            "raw float score delta. See stats.py module docstring for the "
            "binarization justification."
        ),
        "cells": [
            {
                "task": cell.task,
                "length_label": cell.length_label,
                "n_rows": cell.n_rows,
                "comparisons": {
                    name: _comparison_stats_to_dict(comparison_stats)
                    for name, comparison_stats in cell.comparisons.items()
                },
            }
            for cell in cells
        ],
    }
    # Only present when a cpc run was requested (`cpc_run_name` at the
    # `write_ruler_stats`/CLI level) -- omitted entirely otherwise, so the
    # payload shape is byte-identical to before the cpc extension when cpc
    # is not in play.
    if cpc_aggregates is not None:
        payload["cpc_aggregates"] = {
            "note": (
                "Mean bootstrap observed_delta (cpc_score minus the other method's "
                "score) aggregated across tasks, per (comparison, length_label). "
                "'headline_excluding_niah_single_3' is the HEADLINE number -- it "
                "drops the known niah_single_3 dead-zero artifact (see "
                "the published RULER results' Known limitations). 'all_tasks' "
                "includes every task, niah_single_3 included, for full "
                "transparency. Per-task cells above always include niah_single_3 "
                "regardless of aggregate scope."
            ),
            "excluded_task": NIAH_SINGLE_3_TASK,
            "items": [asdict(agg) for agg in cpc_aggregates],
        }
    return payload


def _fmt_p(p: float) -> str:
    return f"{p:.4g}"


def _fmt_ci(result: BootstrapCIResult) -> str:
    return f"{result.observed_delta:+.3f} [{result.ci_low:+.3f}, {result.ci_high:+.3f}]"


def format_markdown_table(
    cells: Sequence[TaskLengthStats],
    *,
    run_name: str,
    cpc_aggregates: Sequence[CpcAggregate] | None = None,
) -> str:
    """One row per (task, length); both named comparisons' McNemar p-value
    and bootstrap CI (on the score delta) side by side."""
    comparison_names = list(cells[0].comparisons.keys()) if cells else [name for name, _, _ in COMPARISONS]

    header_cells = ["task", "length", "n"]
    for name in comparison_names:
        header_cells.append(f"{name} mcnemar p (b,c)")
        header_cells.append(f"{name} bootstrap delta [95% CI]")
    header = "| " + " | ".join(header_cells) + " |"
    separator = "| " + " | ".join(["---"] * len(header_cells)) + " |"

    lines = [
        f"# RULER paired stats -- {run_name}",
        "",
        (
            f"McNemar correctness threshold: score >= {MCNEMAR_CORRECTNESS_THRESHOLD} "
            "(see stats.py module docstring)."
        ),
        "",
        header,
        separator,
    ]
    for cell in cells:
        row_cells = [cell.task, cell.length_label, str(cell.n_rows)]
        for name in comparison_names:
            comparison_stats = cell.comparisons[name]
            mcnemar = comparison_stats.mcnemar
            bootstrap = comparison_stats.bootstrap
            row_cells.append(f"{_fmt_p(mcnemar.p_value)} ({mcnemar.b},{mcnemar.c})")
            row_cells.append(_fmt_ci(bootstrap))
        lines.append("| " + " | ".join(row_cells) + " |")

    if cpc_aggregates:
        lines.extend(
            [
                "",
                "## CPC aggregate summary",
                "",
                (
                    f"Mean bootstrap delta (cpc - other), aggregated across tasks per "
                    f"(comparison, length). `headline_excluding_niah_single_3` is the "
                    f"headline number (excludes the known `{NIAH_SINGLE_3_TASK}` "
                    "dead-zero artifact -- see the published RULER results); "
                    "`all_tasks` includes every task. Per-task rows above always "
                    f"include `{NIAH_SINGLE_3_TASK}` regardless of aggregate scope."
                ),
                "",
                "| comparison | length | scope | n_tasks | mean delta (cpc - other) |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for agg in cpc_aggregates:
            lines.append(
                f"| {agg.comparison} | {agg.length_label} | {agg.scope} | {agg.n_tasks} "
                f"| {agg.mean_observed_delta:+.3f} |"
            )

    return "\n".join(lines) + "\n"


def write_ruler_stats(
    output_root: Path,
    run_name: str,
    tasks: Sequence[str],
    *,
    length_labels: Sequence[str] = DEFAULT_LENGTH_LABELS,
    mcnemar_threshold: float = MCNEMAR_CORRECTNESS_THRESHOLD,
    cpc_run_name: str | None = None,
) -> Tuple[Path, Path]:
    """Compute stats over the real, persisted rows.json files and write both
    a JSON payload and a markdown table to
    `<output_root>/<run_name>_ruler_stats.{json,md}`. Returns
    `(json_path, md_path)`.

    When `cpc_run_name` is given, also joins+computes the three cpc
    comparisons (see `compute_ruler_stats`) and adds a `cpc_aggregates`
    section (see `compute_cpc_aggregates`) to both outputs. Without it
    (the default), output is unchanged from before the cpc extension.
    """
    cells = compute_ruler_stats(
        output_root,
        run_name,
        tasks,
        length_labels=length_labels,
        mcnemar_threshold=mcnemar_threshold,
        cpc_run_name=cpc_run_name,
    )
    cpc_aggregates = compute_cpc_aggregates(cells) if cpc_run_name is not None else None
    json_payload = cells_to_json_payload(cells, run_name=run_name, cpc_aggregates=cpc_aggregates)
    markdown = format_markdown_table(cells, run_name=run_name, cpc_aggregates=cpc_aggregates)

    json_path = output_root / f"{run_name}_ruler_stats.json"
    md_path = output_root / f"{run_name}_ruler_stats.md"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired McNemar + bootstrap CI stats over a real RULER run's rows.json files."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names.")
    parser.add_argument(
        "--length-labels",
        type=str,
        default=",".join(DEFAULT_LENGTH_LABELS),
        help="Comma-separated context length labels (default: 8k,16k).",
    )
    parser.add_argument(
        "--cpc-run-name",
        type=str,
        default=None,
        help=(
            "Optional cpc-arm run name (e.g. official_ruler_full_run_v1_cpc). When given, "
            "joins <output-root>/<cpc-run-name>_<length>/<task>/rows.json onto the main "
            "run's rows by position and additionally computes cpc_vs_needlepath (headline), "
            "cpc_vs_full_context, cpc_vs_llmlingua2, plus aggregate summaries."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    length_labels = [label.strip() for label in args.length_labels.split(",") if label.strip()]
    json_path, md_path = write_ruler_stats(
        args.output_root,
        args.run_name,
        tasks,
        length_labels=length_labels,
        cpc_run_name=args.cpc_run_name,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
