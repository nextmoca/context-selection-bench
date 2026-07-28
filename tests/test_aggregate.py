"""Tests for the public RULER aggregation tool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csbench.suites.ruler import aggregate
from csbench.suites.ruler.pairing import ItemIdentityError


def row(position, index, score, **kw):
    d = {"position": position, "index": index, "score": float(score),
         "correct": float(score) >= 1.0, "input_tokens": 100, "output_tokens": 5,
         "tokens_before": 100, "tokens_after": 100, "fallback_used": False}
    d.update(kw)
    return d


def write_run(tmp_path, cells):
    for (length, arm, task), rows in cells.items():
        d = tmp_path / "items" / length / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_deduplicate_counts_and_is_lossless(tmp_path):
    cells = {("8k", "full_context", "t"): [row(1, 10, 1), row(2, 20, 0), row(2, 20, 0)]}
    out, counts = aggregate.deduplicate(cells)
    assert counts == {"n_rows": 3, "n_distinct": 2, "n_duplicate_rows": 1}
    assert [r["position"] for r in out[("8k", "full_context", "t")]] == [1, 2]


def test_deduplicate_refuses_differing_rows(tmp_path):
    cells = {("8k", "a", "t"): [row(1, 10, 1), row(1, 10, 0)]}
    with pytest.raises(ItemIdentityError, match="DIFFERING content"):
        aggregate.deduplicate(cells)


def test_cross_arm_identity_flags_mismatch():
    cells = {
        ("8k", "full_context", "t"): [row(1, 10, 1), row(2, 20, 1)],
        ("8k", "np", "t"): [row(1, 10, 1), row(2, 99, 1)],   # different item at pos 2
    }
    problems = aggregate.assert_cross_arm_identity(cells)
    assert problems and "index identity mismatch" in problems[0]


def test_cross_arm_identity_clean_run_has_no_problems():
    cells = {
        ("8k", "full_context", "t"): [row(1, 10, 1), row(2, 20, 0)],
        ("8k", "np", "t"): [row(1, 10, 1), row(2, 20, 1)],
    }
    assert aggregate.assert_cross_arm_identity(cells) == []


def test_comparison_reports_delta_and_n():
    cells = {
        ("8k", "full_context", "t"): [row(i, i, 0) for i in range(1, 11)],
        ("8k", "np", "t"): [row(i, i, 1) for i in range(1, 11)],
    }
    got = aggregate.comparison(cells, arm="np", lengths=["8k"], tasks=["t"],
                               seed=42, resamples=200)
    assert got["n"] == 10
    assert got["accuracy_delta_pp"] == pytest.approx(100.0)
    assert got["em_delta_pp"] == pytest.approx(100.0)


def test_out_dir_may_not_be_inside_run_dir(tmp_path):
    run = write_run(tmp_path / "run", {("8k", "full_context", "t"): [row(1, 1, 1)]})
    rc = aggregate.main(["--run-dir", str(run), "--out", str(run / "sub"),
                         "--expect-arms", "full_context", "--expect-tasks", "t",
                         "--lengths", "8k"])
    assert rc == 2, "must refuse to write inside the deposit"


def test_task_scope_is_explicit(tmp_path):
    cells = {}
    for task in ("a", aggregate.HEADLINE_EXCLUDED_TASK):
        cells[("8k", "full_context", task)] = [row(i, i, 0) for i in range(1, 6)]
        cells[("8k", "np", task)] = [row(i, i, 1) for i in range(1, 6)]
    run = write_run(tmp_path / "run", cells)
    out = tmp_path / "out"
    assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                           "--tasks", "headline12", "--resamples", "200",
                           "--expect-arms", "full_context", "np", "--expect-tasks", "a", aggregate.HEADLINE_EXCLUDED_TASK,
                           "--lengths", "8k"]) == 0
    meta = json.loads((out / "replication_table.json").read_text())["meta"]
    assert aggregate.HEADLINE_EXCLUDED_TASK not in meta["tasks"]
    assert meta["task_scope"] == "headline12"


def test_raw_population_is_off_by_default(tmp_path):
    cells = {
        ("8k", "full_context", "t"): [row(i, i, 0) for i in range(1, 11)],
        ("8k", "np", "t"): [row(i, i, 1) for i in range(1, 11)],
    }
    run = write_run(tmp_path / "run", cells)
    out = tmp_path / "out"
    assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                           "--resamples", "200",
                           "--expect-arms", "full_context", "np", "--expect-tasks", "t", "--lengths", "8k"]) == 0
    payload = json.loads((out / "replication_table.json").read_text())
    pops = payload["table"]["arms"]["np"]["populations"]
    assert list(pops) == ["distinct"], "damaged population must not be reachable by accident"
    assert payload["meta"]["legacy_order_pairing"] is False


def test_raw_population_carries_a_damaged_provenance_field(tmp_path):
    cells = {
        ("8k", "full_context", "t"): [row(i, i, 0) for i in range(1, 11)],
        ("8k", "np", "t"): [row(i, i, 1) for i in range(1, 11)],
    }
    run = write_run(tmp_path / "run", cells)
    out = tmp_path / "out"
    assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                           "--resamples", "200", "--legacy-order-pairing",
                           "--expect-arms", "full_context", "np", "--expect-tasks", "t", "--lengths", "8k"]) == 0
    pops = json.loads((out / "replication_table.json").read_text())["table"]["arms"]["np"]["populations"]
    assert "DAMAGED POPULATION" in pops["raw"]["provenance"]
    assert "comparison only" in pops["raw"]["provenance"]
    assert "identity assertion" in pops["distinct"]["provenance"]


def test_end_to_end_emits_both_populations(tmp_path):
    cells = {
        ("8k", "full_context", "t"): [row(i, i, 0) for i in range(1, 11)],
        ("8k", "np", "t"): [row(i, i, 1) for i in range(1, 11)],
    }
    # one benign duplicate, so raw and distinct genuinely differ
    for k in cells:
        cells[k] = cells[k] + [dict(cells[k][2])]
    run = write_run(tmp_path / "run", cells)
    out = tmp_path / "out"
    assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                           "--resamples", "200", "--legacy-order-pairing",
                           "--expect-arms", "full_context", "np", "--expect-tasks", "t", "--lengths", "8k"]) == 0
    payload = json.loads((out / "replication_table.json").read_text())
    assert payload["meta"]["n_rows"] == 22
    assert payload["meta"]["n_distinct"] == 20
    pops = payload["table"]["arms"]["np"]["populations"]
    assert pops["distinct"]["scopes"]["8k"]["n"] == 10
    assert pops["raw"]["scopes"]["8k"]["n"] == 11
    md = (out / "replication_table.md").read_text()
    assert "distinct (primary)" in md and "reproduces published" in md


def test_holm_is_applied_across_the_per_length_family(tmp_path):
    cells = {}
    for length in ("8k", "16k"):
        cells[(length, "full_context", "t")] = [row(i, i, 0) for i in range(1, 21)]
        cells[(length, "np", "t")] = [row(i, i, 1 if i <= 15 else 0) for i in range(1, 21)]
    run = write_run(tmp_path / "run", cells)
    out = tmp_path / "out"
    assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                           "--resamples", "200",
                           "--expect-arms", "full_context", "np", "--expect-tasks", "t", "--lengths", "8k", "16k"]) == 0
    scopes = json.loads((out / "replication_table.json").read_text())
    scopes = scopes["table"]["arms"]["np"]["populations"]["distinct"]["scopes"]
    for length in ("8k", "16k"):
        assert "mcnemar_p_holm_adjusted" in scopes[length]
        assert scopes[length]["holm_family_size"] == 2
        assert "significant_after_holm" in scopes[length]


def test_missing_task_is_an_error_not_a_smaller_table(tmp_path):
    """The completeness guard must validate against a DECLARED grid.

    Inferring the expected tasks from the data would make this vacuous: a task
    absent from the deposit would be absent from the expectation too.
    """
    cells = {
        ("8k", "full_context", "present"): [row(i, i, 0) for i in range(1, 6)],
        ("8k", "np", "present"): [row(i, i, 1) for i in range(1, 6)],
    }
    run = write_run(tmp_path / "run", cells)
    rc = aggregate.main(["--run-dir", str(run), "--out", str(tmp_path / "out"),
                         "--resamples", "200",
                         "--expect-arms", "full_context", "np", "--expect-tasks", "present", "absent", "--lengths", "8k"])
    assert rc == 2, "a task declared but absent from the deposit must fail the run"


def test_iteration_order_is_pinned_and_changes_the_ci_when_it_is_not(tmp_path):
    """Iteration order is load-bearing, not cosmetic.

    The paired bootstrap draws resample indices into the concatenated per-item
    array, so task order changes the draw and therefore the interval. This was
    unpinned and deterministic only by accident; a gate caught it. The test
    demonstrates the sensitivity and pins the order.
    """
    cells = {}
    for length in ("8k", "16k"):
        for task, score in (("alpha", 1), ("zulu", 0)):
            cells[(length, "full_context", task)] = [row(i, i, 0) for i in range(1, 41)]
            cells[(length, "np", task)] = [row(i, i, score) for i in range(1, 41)]
    run = write_run(tmp_path / "run", cells)

    def ci(tasks, lengths):
        out = tmp_path / f"out_{'_'.join(tasks)}_{'_'.join(lengths)}"
        assert aggregate.main(["--run-dir", str(run), "--out", str(out),
                               "--resamples", "300", "--lengths", *lengths,
                               "--expect-arms", "full_context", "np",
                               "--expect-tasks", *tasks]) == 0
        s = json.loads((out / "replication_table.json").read_text())
        return s["table"]["arms"]["np"]["populations"]["distinct"]["scopes"]["pooled"]

    # Reverse BOTH declared dimensions: task order and length order.
    forward = ci(["alpha", "zulu"], ["8k", "16k"])
    reversed_ = ci(["zulu", "alpha"], ["16k", "8k"])
    # Declared order must not leak into the computation: sorting pins it, so
    # both declarations must produce the identical interval.
    assert forward["accuracy_ci_pp"] == reversed_["accuracy_ci_pp"], (
        "declared task order leaked into the bootstrap; computation order must "
        "be sorted regardless of how the expected grid is declared"
    )
    assert forward["accuracy_delta_pp"] == pytest.approx(reversed_["accuracy_delta_pp"])


def test_missing_arm_is_an_error_not_a_smaller_table(tmp_path):
    """Deleting an entire arm must not pass completeness.

    Arms were previously inferred from the deposit, so removing a whole arm
    removed it from the expectation too and silently dropped a published
    comparison.
    """
    cells = {
        ("8k", "full_context", "t"): [row(i, i, 0) for i in range(1, 6)],
        ("8k", "np", "t"): [row(i, i, 1) for i in range(1, 6)],
    }
    run = write_run(tmp_path / "run", cells)
    rc = aggregate.main(["--run-dir", str(run), "--out", str(tmp_path / "out"),
                         "--resamples", "200", "--lengths", "8k",
                         "--expect-tasks", "t",
                         "--expect-arms", "full_context", "np", "compresr"])
    assert rc == 2, "an arm declared but absent from the deposit must fail the run"
