"""Regenerate the RULER replication table from deposited per-item rows.

Public reproduction path. Given a deposit directory of per-item JSONL cells,
this recomputes every published figure family using only public-repo code
(``csbench.stats`` for the inference, ``csbench.suites.ruler.pairing`` for the
item identity contract). No API calls, no dataset download, no arm code.

It exists because the deposit could previously be *verified* against a manifest
but not *recomputed*: the per-cell comparisons lived inside the run harness and
ran only as a side effect of executing a paid benchmark run.

Populations
-----------
The published ``ruler_repl_v1`` cells were damaged by a resume-overwrite defect
(see ``pairing.dedup_benign_copies``): each affected cell holds 100 rows but
fewer than 100 distinct positions, because byte-identical copies of one row
replaced the evaluations they displaced. Those evaluations are unrecoverable.

Two populations are therefore reported side by side, and never mixed:

- ``distinct`` (primary): one row per distinct position. This is what the
  corrected figures are computed on.
- ``raw`` (opt-in only, behind ``--legacy-order-pairing``): every row as stored,
  duplicates retained and aligned by file order rather than by the item identity
  contract. This reproduces the originally published numbers so a reader can see
  exactly what changed, and is never a corrected result. Off by default.

``n_distinct`` is reported beside ``n_rows`` everywhere.

Usage
-----
    python -m csbench.suites.ruler.aggregate \\
        --run-dir deposit/ruler_repl_v1 --out /tmp/table
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from csbench.stats import bootstrap_ci_delta, holm_adjust, mcnemar_test
from csbench.suites.ruler.pairing import (
    ItemIdentityError,
    dedup_benign_copies,
    item_key,
    pair_rows,
)

BASELINE_ARM = "full_context"
DEFAULT_SEED = 42
DEFAULT_RESAMPLES = 10000
HEADLINE_EXCLUDED_TASK = "niah_single_3"

# The grid a complete RULER run must cover. Declared here so `assert_complete_grid`
# validates against an expectation rather than against whatever happens to be on
# disk. Override with --expect-tasks / --lengths for a deliberately partial run.
EXPECTED_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
)
EXPECTED_LENGTHS = ("8k", "16k")
EXPECTED_ARMS = ("full_context", "needlepath", "compresr")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_cells(run_dir: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """``(length, arm, task) -> rows`` from ``<run_dir>/items/*/*/*.jsonl``."""
    items = run_dir / "items"
    if not items.is_dir():
        raise SystemExit(f"error: {items} not found; expected a deposited run directory")
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(items.glob("*/*/*.jsonl")):
        length, arm, task = path.parts[-3], path.parts[-2], path.name[: -len(".jsonl")]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            cells[(length, arm, task)] = rows
    if not cells:
        raise SystemExit(f"error: no per-item rows under {items}")
    return cells


def deduplicate(
    cells: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]]
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, int]]:
    """Collapse byte-identical repeats, refusing to drop differing rows."""
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    n_rows = n_distinct = n_removed = 0
    for key, rows in cells.items():
        deduped, removed = dedup_benign_copies(rows, source="/".join(key))
        out[key] = deduped
        n_rows += len(rows)
        n_distinct += len(deduped)
        n_removed += removed
    return out, {"n_rows": n_rows, "n_distinct": n_distinct, "n_duplicate_rows": n_removed}


def assert_complete_grid(
    cells: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]],
    *, lengths: Sequence[str], tasks: Sequence[str], arms: Sequence[str],
) -> list[str]:
    """Every selected (length, task, arm) cell must be present.

    A missing cell was previously skipped with ``continue``, which silently
    shrank the pooled population and the Holm family. Silently computing on a
    smaller population than requested is the exact defect this tool exists to
    correct, so absence is now an error.
    """
    missing = [
        f"{length}/{arm}/{task}"
        for length in lengths for task in tasks for arm in arms
        if (length, arm, task) not in cells
    ]
    if missing:
        return [f"{len(missing)} selected cell(s) absent: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"]
    return []


def assert_cross_arm_identity(
    cells: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]]
) -> list[str]:
    """Every arm in a (length, task) must carry the identical position sequence.

    Checked over ALL arm pairs, not only against the baseline, so two
    non-baseline arms cannot disagree undetected.
    """
    problems: list[str] = []
    by_cell: dict[tuple[str, str], dict[str, Sequence[dict[str, Any]]]] = defaultdict(dict)
    for (length, arm, task), rows in cells.items():
        by_cell[(length, task)][arm] = rows
    for (length, task), arms in sorted(by_cell.items()):
        names = sorted(arms)
        if len(names) < 2:
            continue
        # No tripwire here: this join reads r["position"] explicitly, so there
        # is no key-field ambiguity to guard against. Historical cells are also
        # legitimately sparse (destroyed positions), so any bound inferred from
        # the surviving rows would false-positive. The tripwire belongs where a
        # key field is actually chosen: the resume cache.
        anchor = BASELINE_ARM if BASELINE_ARM in arms else names[0]
        for arm in names:
            if arm == anchor:
                continue
            try:
                pair_rows(arms[arm], arms[anchor], arm_source=f"{length}/{arm}/{task}",
                          baseline_source=f"{length}/{anchor}/{task}")
            except ItemIdentityError as exc:
                problems.append(str(exc))
    return problems


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _paired(arm_rows, base_rows, *, strict: bool = True):
    """Align two arms' rows.

    ``strict`` uses the position-based identity contract and is correct for any
    sound row set. It cannot be used on the damaged ``raw`` population, whose
    duplicate positions it rightly rejects; there we fall back to the file-order
    alignment that produced the originally published numbers, purely so those
    numbers can be reproduced and compared.
    """
    if strict:
        _, a, b = pair_rows(arm_rows, base_rows)
    else:
        key = lambda r: item_key(r)
        a = sorted(arm_rows, key=key)
        b = sorted(base_rows, key=key)
        if len(a) != len(b):
            raise ItemIdentityError(
                f"raw population row counts differ: {len(a)} vs {len(b)}"
            )
    return (
        [float(r.get("score", 0.0)) for r in a],
        [float(r.get("score", 0.0)) for r in b],
        [bool(r.get("correct")) for r in a],
        [bool(r.get("correct")) for r in b],
    )


def comparison(
    cells: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]],
    *,
    arm: str,
    lengths: Iterable[str],
    tasks: Iterable[str],
    seed: int,
    resamples: int,
    strict: bool = True,
) -> dict[str, Any] | None:
    """Pooled figures for one arm over a (length, task) scope."""
    sa: list[float] = []
    sb: list[float] = []
    ea: list[bool] = []
    eb: list[bool] = []
    for length in lengths:
        for task in tasks:
            arm_rows = cells.get((length, arm, task))
            base_rows = cells.get((length, BASELINE_ARM, task))
            if not arm_rows or not base_rows:
                raise ItemIdentityError(
                    f"{length}/{task}: missing rows for "
                    f"{arm if not arm_rows else BASELINE_ARM}; refusing to pool a "
                    "population smaller than the one requested"
                )
            x_s, y_s, x_e, y_e = _paired(arm_rows, base_rows, strict=strict)
            sa += x_s; sb += y_s; ea += x_e; eb += y_e
    if not sa:
        return None
    acc = bootstrap_ci_delta(sa, sb, n_resamples=resamples, seed=seed)
    em = bootstrap_ci_delta(ea, eb, n_resamples=resamples, seed=seed)
    mc = mcnemar_test(ea, eb)
    return {
        "n": len(sa),
        "accuracy_delta_pp": acc.observed_delta * 100.0,
        "accuracy_ci_pp": [acc.ci_low * 100.0, acc.ci_high * 100.0],
        "em_delta_pp": em.observed_delta * 100.0,
        "em_ci_pp": [em.ci_low * 100.0, em.ci_high * 100.0],
        "mcnemar_p_raw": mc.p_value,
        "mcnemar_method": mc.method,
        "mcnemar_b": mc.b,
        "mcnemar_c": mc.c,
        "token_reduction_pct": None,
        "mean_input_tokens": None,
        "fallback_rate": None,
    }


def arm_descriptives(
    cells: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]],
    *,
    arm: str,
    lengths: Iterable[str],
    tasks: Iterable[str],
    usd_per_1m_blended: float | None,
    service_fee_usd: float = 0.0,
) -> dict[str, Any]:
    """Token, fallback and cost descriptives on one consistent population.

    The published cost figures were computed on inconsistent populations (one
    arm on raw totals, another on deduplicated), so they are recomputed here on
    whichever population the caller passed.
    """
    rows: list[dict[str, Any]] = []
    for length in lengths:
        for task in tasks:
            rows += list(cells.get((length, arm, task), []))
    if not rows:
        return {}
    # Baseline rows over the same scope, for the billed-basis reduction.
    base_rows: list[dict[str, Any]] = []
    for length in lengths:
        for task in tasks:
            base_rows += list(cells.get((length, BASELINE_ARM, task), []))

    n = len(rows)
    tin = sum(float(r.get("input_tokens", 0.0)) for r in rows)
    tout = sum(float(r.get("output_tokens", 0.0)) for r in rows)
    before = sum(float(r.get("tokens_before", 0.0)) for r in rows)
    after = sum(float(r.get("tokens_after", 0.0)) for r in rows)
    base_tin = sum(float(r.get("input_tokens", 0.0)) for r in base_rows)

    out = {
        "n_rows": n,
        "mean_input_tokens": tin / n,
        "mean_output_tokens": tout / n,
        "total_input_tokens": int(tin),
        "total_output_tokens": int(tout),
        # BILLED basis: mean input tokens against the matched full-context arm.
        # This is the figure the paper reports. The arm's internal
        # tokens_before/tokens_after ratio is a different quantity (it ignores
        # the prompt scaffold and the fallback substitution) and is reported
        # separately so the two are never confused.
        "input_token_reduction_pct": (
            (base_tin - tin) / base_tin * 100.0 if base_tin and arm != BASELINE_ARM else
            (0.0 if arm == BASELINE_ARM else None)
        ),
        "internal_selection_reduction_pct": ((before - after) / before * 100.0) if before else None,
        "fallback_rate": sum(1 for r in rows if r.get("fallback_used")) / n,
    }
    if usd_per_1m_blended is not None:
        api = (tin + tout) / 1e6 * usd_per_1m_blended
        out["api_cost_usd"] = api
        out["service_fee_usd"] = service_fee_usd
        out["cost_usd"] = api + service_fee_usd
        out["cost_basis_usd_per_1m_blended"] = usd_per_1m_blended
    return out


def build_table(
    cells_raw, cells_distinct, *, lengths, tasks, seed, resamples, usd_per_1m,
    legacy_order_pairing: bool = False, service_fees: dict[str, float] | None = None
) -> dict[str, Any]:
    arms = sorted({a for (_, a, _) in cells_distinct} - {BASELINE_ARM})
    table: dict[str, Any] = {"arms": {}}
    populations = [("distinct", cells_distinct)]
    if legacy_order_pairing:
        populations.append(("raw", cells_raw))
    for arm in arms:
        entry: dict[str, Any] = {"populations": {}}
        for label, cells in populations:
            scopes: dict[str, Any] = {}
            for scope, ls in [(l, [l]) for l in lengths] + [("pooled", list(lengths))]:
                got = comparison(cells, arm=arm, lengths=ls, tasks=tasks,
                                 seed=seed, resamples=resamples,
                                 strict=(label == "distinct"))
                if got:
                    scopes[scope] = got
            # Holm over the co-primary per-length family
            # Family is the requested per-length tests. Deriving it from
            # whichever scopes happened to survive would silently weaken the
            # correction from alpha/2 to alpha/1.
            missing_scopes = [l for l in lengths if l not in scopes]
            if missing_scopes:
                raise ItemIdentityError(
                    f"{arm}: per-length scope(s) {missing_scopes} absent; the Holm "
                    "family would be under-sized and the correction too weak"
                )
            family = {l: scopes[l]["mcnemar_p_raw"] for l in lengths}
            if family:
                for r in holm_adjust(family):
                    scopes[r.label]["mcnemar_p_holm_adjusted"] = r.p_adjusted
                    scopes[r.label]["holm_threshold"] = r.threshold
                    scopes[r.label]["holm_family_size"] = len(family)
                    scopes[r.label]["significant_after_holm"] = r.reject
            entry["populations"][label] = {
                "provenance": (
                    "distinct positions, position-based pairing with an index "
                    "identity assertion" if label == "distinct" else
                    "DAMAGED POPULATION: duplicate rows retained and aligned by file "
                    "order, not by the item identity contract. Produced under "
                    "--legacy-order-pairing to reproduce the originally published "
                    "figures for comparison only. Not a corrected result."
                ),
                "scopes": scopes,
                "descriptives": {
                    a: arm_descriptives(cells, arm=a, lengths=lengths, tasks=tasks,
                                        usd_per_1m_blended=usd_per_1m,
                                        service_fee_usd=(service_fees or {}).get(a, 0.0))
                    for a in (arm, BASELINE_ARM)
                },
            }
        table["arms"][arm] = entry
    return table


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_markdown(payload: Mapping[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# RULER replication table (regenerated from deposited per-item rows)",
        "",
        f"Run: `{meta['run_dir']}`  ",
        f"Generated: {meta['generated_at']}  ",
        f"Tasks: {meta['task_scope']} ({len(meta['tasks'])})  Lengths: {', '.join(meta['lengths'])}  ",
        f"Bootstrap: {meta['resamples']} resamples, seed {meta['seed']}",
        "",
        f"Rows on disk: {meta['n_rows']}  |  distinct positions: {meta['n_distinct']}  "
        f"|  duplicate rows: {meta['n_duplicate_rows']}",
        "",
    ]
    if meta["n_duplicate_rows"]:
        lines += [
            "> The `distinct` population is primary. Duplicate rows are byte-identical",
            "> copies written by a resume-overwrite defect; the evaluations they replaced",
            "> are unrecoverable. The `raw` population is shown only to reproduce the",
            "> originally published figures.",
            "",
        ]
    for arm, entry in payload["table"]["arms"].items():
        lines += [f"## {arm} vs {BASELINE_ARM}", ""]
        for pop in ("distinct", "raw"):
            block = entry["populations"].get(pop)
            if not block:
                continue
            tag = "primary" if pop == "distinct" else "secondary, reproduces published"
            lines += [
                f"### population: {pop} ({tag})",
                "",
                "| scope | n | acc delta | acc 95% CI | EM delta | EM 95% CI | McNemar p | Holm p | sig |",
                "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
            ]
            for scope, s in block["scopes"].items():
                holm = s.get("mcnemar_p_holm_adjusted")
                sig = s.get("significant_after_holm")
                lines.append(
                    f"| {scope} | {s['n']} | {s['accuracy_delta_pp']:+.2f}pp | "
                    f"[{s['accuracy_ci_pp'][0]:+.2f}, {s['accuracy_ci_pp'][1]:+.2f}] | "
                    f"{s['em_delta_pp']:+.2f}pp | "
                    f"[{s['em_ci_pp'][0]:+.2f}, {s['em_ci_pp'][1]:+.2f}] | "
                    f"{s['mcnemar_p_raw']:.4f} | "
                    f"{'' if holm is None else f'{holm:.4f}'} | "
                    f"{'' if sig is None else ('yes' if sig else 'no')} |"
                )
            lines.append("")
            lines += [
                "| arm | rows | mean input tokens | billed reduction | internal sel. ratio "
                "| fallback rate | cost |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for a, d in block["descriptives"].items():
                if not d:
                    continue
                red = d.get("input_token_reduction_pct")
                internal = d.get("internal_selection_reduction_pct")
                cost = d.get("cost_usd")
                fee = d.get("service_fee_usd") or 0.0
                cost_s = "" if cost is None else (
                    f"${cost:.2f}" + (f" (incl. ${fee:.2f} fee)" if fee else "")
                )
                lines.append(
                    f"| {a} | {d['n_rows']} | {d['mean_input_tokens']:.0f} | "
                    f"{'' if red is None else f'{red:.2f}%'} | "
                    f"{'' if internal is None else f'{internal:.2f}%'} | "
                    f"{d['fallback_rate']:.1%} | {cost_s} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory; never the deposit directory")
    ap.add_argument("--tasks", choices=("all", "headline12"), default="all",
                    help="'all' = every task present; 'headline12' excludes "
                         f"{HEADLINE_EXCLUDED_TASK}. No hidden default.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--service-fee", nargs="+", default=None, metavar="ARM=USD",
                    help="flat non-token charge for an arm, e.g. compresr=2.68; "
                         "some arms bill a service fee on top of model tokens and "
                         "a single blended token rate cannot reproduce their cost")
    ap.add_argument("--expect-tasks", nargs="+", default=None,
                    help="declared task set the run must cover in full "
                         "(default: the 13 official RULER tasks)")
    ap.add_argument("--expect-arms", nargs="+", default=None,
                    help="declared arms the run must cover in full "
                         "(default: full_context needlepath compresr)")
    ap.add_argument("--lengths", nargs="+", default=None,
                    help="declared lengths the run must cover in full (default: 8k 16k)")
    ap.add_argument("--legacy-order-pairing", action="store_true",
                    help="ALSO emit the damaged raw population, aligned by file order "
                         "rather than by item identity, to reproduce the originally "
                         "published figures. Off by default. Never a corrected result.")
    ap.add_argument("--usd-per-1m-blended", type=float, default=None,
                    help="blended rate for the cost column; omit to leave cost blank")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    out = args.out.resolve()
    if out == run_dir or run_dir in out.parents:
        return _fail("--out must not be inside --run-dir; this tool never writes to a deposit")

    service_fees: dict[str, float] = {}
    for spec in (args.service_fee or []):
        if "=" not in spec:
            return _fail(f"--service-fee expects ARM=USD, got {spec!r}")
        k, v = spec.split("=", 1)
        try:
            service_fees[k] = float(v)
        except ValueError:
            return _fail(f"--service-fee value is not a number: {spec!r}")

    cells_raw = load_cells(run_dir)
    cells_distinct, counts = deduplicate(cells_raw)

    # The expected grid is declared, NOT inferred from the data being validated.
    # Inferring it would make the completeness check vacuous: a task directory
    # missing from the deposit would simply be absent from the expected set and
    # the check would pass on a smaller table.
    # Two orders, deliberately separated.
    #
    # The DECLARED grid is what completeness is validated against, and it is
    # never inferred from the data being validated.
    declared_lengths = list(args.lengths or EXPECTED_LENGTHS)
    declared_tasks = [t for t in (args.expect_tasks or EXPECTED_TASKS)
                      if not (args.tasks == "headline12" and t == HEADLINE_EXCLUDED_TASK)]
    #
    # The COMPUTATION order is sorted, and that is load-bearing rather than
    # cosmetic. The paired bootstrap draws resample indices into the
    # concatenated per-item array, so iteration order changes the draw and
    # therefore the interval. Nothing pinned this before; the tool was
    # deterministic by accident. Sorting pins it, and matches the order under
    # which the published intervals were computed.
    lengths = sorted(declared_lengths, key=lambda s: int(s.rstrip("k")))
    tasks = sorted(declared_tasks)
    # Arms are declared for the same reason tasks and lengths are: inferring
    # them from the deposit would let a whole arm be deleted and the run still
    # report completeness, silently removing a published comparison.
    declared_arms = list(args.expect_arms or EXPECTED_ARMS)
    if BASELINE_ARM not in declared_arms:
        return _fail(f"declared arms must include the baseline {BASELINE_ARM!r}")
    problems = assert_complete_grid(cells_distinct, lengths=declared_lengths,
                                    tasks=declared_tasks, arms=declared_arms)
    problems += assert_cross_arm_identity(cells_distinct)
    if problems:
        return _fail(
            "cross-arm item identity failed; refusing to compute a paired statistic:\n  "
            + "\n  ".join(problems[:5])
        )

    if args.legacy_order_pairing:
        print("=" * 78, file=sys.stderr)
        print("WARNING: --legacy-order-pairing is ON.", file=sys.stderr)
        print("Also emitting the DAMAGED population: duplicate rows retained and", file=sys.stderr)
        print("aligned by file order, not by the item identity contract. Those", file=sys.stderr)
        print("figures reproduce the originally published numbers for comparison", file=sys.stderr)
        print("only. They are NOT a corrected result. The primary population", file=sys.stderr)
        print("remains 'distinct'.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)

    table = build_table(cells_raw, cells_distinct, lengths=lengths, tasks=tasks,
                        seed=args.seed, resamples=args.resamples,
                        usd_per_1m=args.usd_per_1m_blended,
                        legacy_order_pairing=args.legacy_order_pairing,
                        service_fees=service_fees)

    payload = {
        "meta": {
            "run_dir": str(run_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lengths": lengths,
            "tasks": tasks,
            "task_scope": args.tasks,
            "iteration_order": "sorted (pins the bootstrap resample draw)",
            "legacy_order_pairing": bool(args.legacy_order_pairing),
            "seed": args.seed,
            "resamples": args.resamples,
            **counts,
        },
        "table": table,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "replication_table.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out}/replication_table.json and .md", file=sys.stderr)
    print(f"rows={counts['n_rows']} distinct={counts['n_distinct']} "
          f"duplicates={counts['n_duplicate_rows']}", file=sys.stderr)
    return 0


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
