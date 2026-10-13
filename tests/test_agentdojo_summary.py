"""Summarizer: gates, directional-only handling, metric arithmetic, attribution."""

from __future__ import annotations

import json

import pytest

from agentdojo_fakes import FakeLauncher, default_outcomes, fake_environment, small_config
from csbench.suites.agentdojo import ARM_BASELINE, ARM_NEEDLEPATH
from csbench.suites.agentdojo.runner import Runner
from csbench.suites.agentdojo.summarize import (
    CONTEXT_REDUCTION_LABEL,
    SummaryGateError,
    percentile,
    summarize_run,
)

_stamps = iter(f"2026102{i:02d}T000000Z" for i in range(100))


def complete_run(tmp_path, repeats=2, planned=2, plan=None, outcomes=default_outcomes):
    cfg = small_config(repeats=planned)
    launcher = FakeLauncher(cfg, plan=plan, outcomes=outcomes)
    runner = Runner(cfg, tmp_path / "run", launcher=launcher, environment=fake_environment(), stamp=lambda: next(_stamps))
    runner.start()
    for r in range(1, repeats + 1):
        for arm in (ARM_BASELINE, ARM_NEEDLEPATH):
            runner.run_arm(r, arm)
    runner.manifest.save(runner.manifest_path)
    return runner


def test_summary_of_a_complete_run_has_the_expected_arithmetic(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    s = summarize_run(runner.run_root)
    assert s["evidence_class"] == "complete" and "complete" in s["statement"]
    rep = s["per_repeat"]["1"]
    # fake outcomes: baseline 4/4 utility, 2/4 attacks; selector 3/4 utility, 0/4 attacks
    assert rep["baseline"] == {**rep["baseline"], "rows": 4, "utility_count": 4, "attack_success_count": 2}
    assert rep["baseline"]["attack_success_rate"] == 0.5 and rep["baseline"]["attack_resistance"] == 0.5
    sel = rep["needlepath_r6_text_autochunk"]
    assert sel["utility_count"] == 3 and sel["attack_success_count"] == 0 and sel["attack_resistance"] == 1.0
    assert rep["delta"] == {
        "utility_delta": 0.75 - 1.0,
        "asr_reduction": 0.5,
        "resistance_delta": 0.5,
        "counts": {"baseline_utility": 4, "r6_utility": 3, "baseline_attack_success": 2, "r6_attack_success": 0, "rows": 4},
    }
    # two applied events (100 -> 40) and two pass-throughs (100 -> 100): 400 -> 280
    red = rep[CONTEXT_REDUCTION_LABEL]
    assert red["label"] == "intermediate tool-result context reduction"
    assert (red["original_tool_context_tokens"], red["delivered_tool_context_tokens"]) == (400, 280)
    assert red["reduction"] == pytest.approx(0.3)
    assert red["applied_events"] == 2 and red["passthrough_events"] == 2
    lat = rep["selector_latency"]
    assert lat["n"] == 4 and lat["p50_ms"] == 100.0 and lat["p95_ms"] == 100.0 and lat["mean_ms"] == 100.0
    agg = s["aggregate"]
    assert agg["n_complete_repeats"] == 2 and agg["complete_repeats"] == [1, 2]
    assert agg["utility_delta"]["per_repeat"] == [-0.25, -0.25] and agg["asr_reduction"]["mean"] == 0.5
    assert agg["counts"]["rows"] == 8 and agg["counts"]["r6_attack_success"] == 0
    assert agg[CONTEXT_REDUCTION_LABEL]["reduction"] == pytest.approx(0.3)
    assert "bootstrap_ci95" in agg
    assert "Average security" not in json.dumps(s)
    # artefacts
    out = runner.run_root / "summary"
    assert json.loads((out / "summary.json").read_text())["run_id"] == "run-test"
    cases = [json.loads(l) for l in (out / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 8 and {c["repeat"] for c in cases} == {1, 2}
    c0 = next(c for c in cases if c["user_task_id"] == "user_task_0" and c["injection_task_id"] == "injection_task_0" and c["repeat"] == 1)
    assert c0["selection_applied"] is True and c0[ARM_NEEDLEPATH]["selector"]["request_ids"] == [c0[ARM_NEEDLEPATH]["case_id"] + "/t0"]
    assert c0[ARM_BASELINE]["attack_success"] is True and c0[ARM_NEEDLEPATH]["attack_success"] is False


def test_attribution_crosstab_splits_outcomes_by_applied_selection(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    s = summarize_run(runner.run_root, write=False)
    att = s["aggregate"]["attribution"]
    # applied on injection_task_0 rows (4 across 2 repeats): baseline attacks succeed there, selector none
    assert att["selection_applied"]["cases"] == 4
    assert att["selection_applied"]["baseline_attack_success_count"] == 4
    assert att["selection_applied"]["r6_attack_success_count"] == 0
    assert att["no_selection_applied"]["cases"] == 4
    assert att["no_selection_applied"]["r6_utility_count"] == 2  # user_task_1/injection_task_1 lost in both repeats
    assert "conditional" in att["note"]


def test_summary_refuses_fewer_than_planned_repeats_unless_directional(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=5)
    with pytest.raises(SummaryGateError, match="2 of 5 planned repeats"):
        summarize_run(runner.run_root)
    s = summarize_run(runner.run_root, allow_directional=True, write=False)
    assert s["evidence_class"] == "directional_only"
    assert "Directional evidence only: 2 of 5" in s["statement"]
    assert "bootstrap_ci95" not in s["aggregate"], "no confidence interval on directional evidence"


def test_summary_refuses_a_single_repeat_even_when_directional(tmp_path):
    runner = complete_run(tmp_path, repeats=1, planned=5)
    with pytest.raises(SummaryGateError):
        summarize_run(runner.run_root, allow_directional=True)


def test_summary_refuses_pair_mismatch(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    # tamper: the selector arm of repeat 2 loses a row after acceptance
    arm_dir = runner.config.arm_dir(runner.run_root, 2, ARM_NEEDLEPATH)
    victim = sorted(arm_dir.rglob("injection_task_1.json"))[0]
    victim.unlink()
    with pytest.raises(SummaryGateError, match="missing_case_ids:1"):
        summarize_run(runner.run_root)


def test_summary_refuses_when_an_accepted_arm_lost_its_telemetry(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    arm_dir = runner.config.arm_dir(runner.run_root, 1, ARM_NEEDLEPATH)
    for p in arm_dir.glob("needlepath_metrics_shard_*.jsonl"):
        p.unlink()
    with pytest.raises(SummaryGateError, match="no_selector_telemetry"):
        summarize_run(runner.run_root)


def test_summary_refuses_nonzero_exit_recorded_in_manifest(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    m = json.loads(runner.manifest_path.read_text())
    for a in m["attempts"]:
        if a["arm"] == ARM_BASELINE and a["repeat"] == 1:
            a["shard_exit_codes"]["0"] = 1
    runner.manifest_path.write_text(json.dumps(m))
    with pytest.raises(SummaryGateError, match="exit status"):
        summarize_run(runner.run_root)


def test_summary_refuses_an_incomplete_manifest(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2)
    m = json.loads(runner.manifest_path.read_text())
    m["config"]["selector"]["snapshot"] = None
    runner.manifest_path.write_text(json.dumps(m))
    with pytest.raises(SummaryGateError, match="selector.snapshot"):
        summarize_run(runner.run_root)


def test_archived_attempts_are_outside_discovery(tmp_path):
    runner = complete_run(tmp_path, repeats=2, planned=2, plan={(1, ARM_NEEDLEPATH, 1): "drop_one"})
    failed = [p for p in (runner.run_root / "repeat_1").iterdir() if ".failed_attempt_" in p.name]
    assert len(failed) == 1
    s = summarize_run(runner.run_root, write=False)
    assert s["per_repeat"]["1"]["needlepath_r6_text_autochunk"]["rows"] == 4
    assert s["per_repeat"]["1"][CONTEXT_REDUCTION_LABEL]["events"] == 4, "archived telemetry is not counted"


def test_percentile_is_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([float(i) for i in range(1, 101)], 0.95) == 95.0
