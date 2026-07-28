"""Single shared item-identity and pairing surface for RULER row sets.

Every path that joins, resumes, or persists RULER rows must route through this
module. It exists because the same defect was found and fixed twice:

1. ``ruler_stats.py`` originally aligned rows by RULER's ``index`` field on the
   assumption that alignment was "trivially guaranteed by construction". That
   held for synthetic fixtures and was false for the real dataset, where a
   task's 100 sampled rows contain genuine duplicate ``index`` values. It was
   fixed locally, in that one module.
2. The fix did not generalise. ``ruler/harness.py`` later re-derived the same
   wrong assumption on two independent lines (an ``index``-keyed resume cache
   and an ``index``-keyed comparison join). The resume cache bound one row
   object to several positions, so writing the cell emitted copies of the
   surviving twin and silently destroyed the overwritten evaluations.

The lesson is that a local fix to one module does not prevent a second module
from making the same mistake. So the correct key lives here, once, and the
tripwire below makes the wrong key fail loudly wherever it is attempted.

``position`` (the harness's own 1-based loop counter) is the identity field.
``index`` is RULER's generator-side sample identifier and is NOT unique within
a task; it may be carried as a cross-check but must never be a join key.
"""
from __future__ import annotations

from typing import Any, Hashable, Iterable, Mapping, Sequence


class ItemIdentityError(ValueError):
    """A row set violates the item-identity contract."""


def item_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """The composite identity of one scored row: ``(position, index)``.

    ``position`` alone is the unique key; ``index`` rides along so that a
    same-position-different-index pair is detectable as a distinct fault.
    """
    if row.get("position") is None:
        raise ItemIdentityError(f"row missing required 'position' field: {row!r}")
    if row.get("index") is None:
        raise ItemIdentityError(f"row missing required 'index' field: {row!r}")
    return int(row["position"]), int(row["index"])


def forbid_index_keying(keys: Iterable[Hashable], *, source: str, n: int) -> None:
    """Tripwire: refuse a key set that was not built from ``position``.

    Call this wherever rows are keyed for a join or a resume cache. Positions
    are 1-based and bounded by the number of examples in the cell, so a valid
    key set is a subset of ``1..n``. RULER's ``index`` values are generator-side
    sample ids far outside that range.

    ``n`` is required rather than inferred. An earlier version guessed the bound
    from the key set itself, which rejected legitimate sparse runs (a targeted
    re-run of positions 90-100 looks like sparse large keys) and could accept a
    small index set. Guessing the bound is exactly the class of error this
    module exists to prevent, so the caller must state it.
    """
    ks_all = list(keys)
    if not ks_all:
        return
    # A guard that silently ignores keys of the wrong type is not a guard: a
    # cache keyed on str(index) would pass by having nothing to check.
    non_int = [k for k in ks_all if not isinstance(k, int) or isinstance(k, bool)]
    if non_int:
        raise ItemIdentityError(
            f"{source}: {len(non_int)} non-integer key(s), e.g. {non_int[:5]!r}. "
            "Positions are 1-based integers; a key of any other type means the "
            "row set was keyed on something else."
        )
    ks = ks_all
    out_of_range = sorted(k for k in ks if k < 1 or k > n)
    if out_of_range:
        raise ItemIdentityError(
            f"{source}: {len(out_of_range)} key(s) outside the valid position "
            f"range 1..{n}, e.g. {out_of_range[:5]}. These look like RULER "
            "'index' values, which are not unique within a task and must never "
            "be used as a join or resume key. Key on 'position' via "
            "csbench.suites.ruler.pairing."
        )


def assert_item_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    expected_n: int | None = None,
) -> None:
    """Assert a persisted row set is exactly one row per position, 1..n.

    This is the check that would have caught the 2026-07-17 replication defect
    at write time. It fails on the published ``ruler_repl_v1`` artifacts, which
    is the point.

    Raises:
        ItemIdentityError: on a duplicate ``position``, a missing ``position``
            in the 1..n range, or a row count that disagrees with ``expected_n``.
    """
    seen: dict[int, int] = {}
    for row in rows:
        position, index = item_key(row)
        if position in seen:
            raise ItemIdentityError(
                f"{source}: duplicate position {position} (indices {seen[position]} "
                f"and {index}). A duplicated position means an evaluation was "
                "overwritten by a copy of another row, not merely repeated."
            )
        seen[position] = index

    n = expected_n if expected_n is not None else len(rows)
    missing = sorted(set(range(1, n + 1)) - set(seen))
    if missing:
        raise ItemIdentityError(
            f"{source}: {len(missing)} of {n} positions absent: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. Expected exactly one row per "
            "position 1..n."
        )
    if expected_n is not None and len(rows) != expected_n:
        raise ItemIdentityError(
            f"{source}: {len(rows)} rows for expected_n={expected_n}."
        )


def index_by_position(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> dict[int, Mapping[str, Any]]:
    """``{position: row}``, failing loudly on a duplicate position."""
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        position, _ = item_key(row)
        if position in out:
            raise ItemIdentityError(f"{source}: duplicate position {position}")
        out[position] = row
    return out


def pair_rows(
    arm_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    arm_source: str = "arm",
    baseline_source: str = "baseline",
) -> tuple[list[int], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Pair two arms' rows by ``position``, asserting full identity agreement.

    This is the ONLY sanctioned way to pair RULER rows. It does not fall back to
    an intersection: a position present on one side only, or a position whose
    ``index`` disagrees across arms, is a hard error. A shrinking ``n_paired``
    is a defect to surface, never a population to quietly compute on.

    Returns ``(positions, arm_rows_aligned, baseline_rows_aligned)``, all sorted
    ascending by position and mutually aligned.
    """
    a = index_by_position(arm_rows, source=arm_source)
    b = index_by_position(baseline_rows, source=baseline_source)

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        raise ItemIdentityError(
            f"position sets differ: only in {arm_source}: {only_a[:10]}; "
            f"only in {baseline_source}: {only_b[:10]}"
        )

    positions = sorted(a)
    for p in positions:
        if int(a[p]["index"]) != int(b[p]["index"]):
            raise ItemIdentityError(
                f"index identity mismatch at position {p}: {arm_source} carries "
                f"index {a[p]['index']}, {baseline_source} carries {b[p]['index']}. "
                "The two arms evaluated different underlying items at this position."
            )
    return positions, [a[p] for p in positions], [b[p] for p in positions]


def dedup_benign_copies(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> tuple[list[Mapping[str, Any]], int]:
    """Collapse byte-identical repeats of the same ``(position, index)``.

    FOR READING DAMAGED HISTORICAL ARTIFACTS ONLY. Never use on a fresh run.

    A repeated key whose copies are byte-identical is the fingerprint of the
    resume-overwrite defect: one row object was bound to several positions and
    serialised more than once. Collapsing the copies recovers the surviving
    evaluations, but does NOT recover the evaluations they overwrote, which were
    destroyed before any write. Callers must treat the result as a reduced
    population and report the shortfall.

    Returns ``(deduplicated_rows, n_copies_removed)``.

    Raises:
        ItemIdentityError: if repeated copies are not identical, which would
            mean genuinely different evaluations collided and no safe recovery
            exists.
    """
    import json

    first: dict[tuple[int, int], Mapping[str, Any]] = {}
    order: list[tuple[int, int]] = []
    removed = 0
    for row in rows:
        key = item_key(row)
        if key in first:
            if json.dumps(first[key], sort_keys=True) != json.dumps(row, sort_keys=True):
                raise ItemIdentityError(
                    f"{source}: position {key[0]} repeats with DIFFERING content. "
                    "These are distinct evaluations, not copies; refusing to "
                    "silently drop one."
                )
            removed += 1
            continue
        first[key] = row
        order.append(key)
    return [first[k] for k in order], removed
