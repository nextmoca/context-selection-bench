"""Runner and collector: row completeness, retry isolation, whole-arm validity."""

from __future__ import annotations

import json

import pytest

from agentdojo_fakes import (
    ATTACK,
    SUITE,
    FakeLauncher,
    default_outcomes,
    fake_environment,
    small_config,
    write_complete_arm,
    write_result_file,
)
from csbench.suites.agentdojo import ARM_BASELINE, ARM_NEEDLEPATH
from csbench.suites.agentdojo.collect import (
    CaseRow,
    PairMismatch,
    collect_arm,
    discover_result_files,
    validate_arm,
    validate_pair,
)
from csbench.suites.agentdojo.config import RunManifest
from csbench.suites.agentdojo.ids import case_id, expected_case_ids, pair_id
from csbench.suites.agentdojo.runner import ArmFailed, Runner, archive_failed_attempt

_stamps = iter(f"2026101{i:02d}T000000Z" for i in range(100))


def make_runner(tmp_path, config, launcher):
    return Runner(config, tmp_path / "run", launcher=launcher, environment=fake_environment(), stamp=lambda: next(_stamps))


# --------------------------------------------------------- row completeness


def _rows(ids):
    out = []
    for cid in ids:
        _, _, ut, it, rep, arm = cid.split("/")
        out.append(CaseRow(cid, "/".join(cid.split("/")[:5]), SUITE, ATTACK, ut, it, int(rep[1:]), arm, True, False, None, 1.0, 1, 1, "final_answer"))
    return out


def test_row_completeness_559_of_560_fails_the_arm():
    ut = [f"user_task_{i}" for i in range(40)]
    it = [f"injection_task_{i}" for i in range(14)]
    expected = expected_case_ids(SUITE, ATTACK, ut, it, 1, ARM_BASELINE)
    assert len(expected) == 560
    ok = validate_arm(_rows(sorted(expected)), expected_ids=expected, shard_exit_codes={str(i): 0 for i in range(8)}, require_telemetry=False, events=[])
    assert ok.valid and ok.row_count == 560
    short = validate_arm(_rows(sorted(expected)[:-1]), expected_ids=expected, shard_exit_codes={str(i): 0 for i in range(8)}, require_telemetry=False, events=[])
    assert not short.valid and short.reason == "missing_case_ids:1" and short.row_count == 559


def test_duplicate_unexpected_and_exit_code_gates():
    expected = expected_case_ids(SUITE, ATTACK, ["user_task_0"], ["injection_task_0", "injection_task_1"], 1, ARM_BASELINE)
    rows = _rows(sorted(expected))
    dup = validate_arm(rows + rows[:1], expected_ids=expected, shard_exit_codes={"0": 0}, require_telemetry=False, events=[])
    assert dup.reason == "duplicate_case_ids:1"
    extra = validate_arm(rows + _rows([case_id(SUITE, ATTACK, "user_task_9", "injection_task_0", 1, ARM_BASELINE)]), expected_ids=expected, shard_exit_codes={"0": 0}, require_telemetry=False, events=[])
    assert extra.reason == "unexpected_case_ids:1"
    exit1 = validate_arm(rows, expected_ids=expected, shard_exit_codes={"0": 0, "1": 137}, require_telemetry=False, events=[])
    assert exit1.reason == "shard_exit_nonzero:['1']"
    none = validate_arm(rows, expected_ids=expected, shard_exit_codes={}, require_telemetry=False, events=[])
    assert none.reason == "no_shard_exit_status_recorded"
    rows[0].error = "ServerError: 504"
    err = validate_arm(rows, expected_ids=expected, shard_exit_codes={"0": 0}, require_telemetry=False, events=[])
    assert err.reason == "provider_error_rows:1", "a provider-skipped case is never a scored outcome"


def test_selector_arm_requires_well_formed_telemetry():
    expected = expected_case_ids(SUITE, ATTACK, ["user_task_0"], ["injection_task_0"], 1, ARM_NEEDLEPATH)
    rows = _rows(sorted(expected))
    v = validate_arm(rows, expected_ids=expected, shard_exit_codes={"0": 0}, require_telemetry=True, events=[])
    assert v.reason == "no_selector_telemetry"
    ev = {"event": "needlepath_applied", "operating_point": None, "ingress": "text", "original_token_estimate": 10}
    v = validate_arm(rows, expected_ids=expected, shard_exit_codes={"0": 0}, require_telemetry=True, events=[ev])
    assert v.reason == "telemetry_event_missing:operating_point"


# ------------------------------------------------------------- collector


def test_collector_reads_only_scored_rows_and_joins_telemetry(tmp_path):
    cfg = small_config()
    arm_dir = tmp_path / "repeat_1" / ARM_NEEDLEPATH
    write_complete_arm(arm_dir, cfg, repeat=1, arm=ARM_NEEDLEPATH, outcomes=default_outcomes)
    # preflight artefact and a stray root json must not become rows
    (arm_dir / "gemini-2.5-flash-preview-04-17" / SUITE / "injection_task_0" / "none").mkdir(parents=True)
    (arm_dir / "gemini-2.5-flash-preview-04-17" / SUITE / "injection_task_0" / "none" / "none.json").write_text("{}")
    (arm_dir / "timing.json").write_text("{}")
    assert len(discover_result_files(arm_dir)) == 4
    rows, events = collect_arm(arm_dir, repeat=1, arm=ARM_NEEDLEPATH)
    assert len(rows) == 4 and len(events) == 4
    by = {(r.user_task_id, r.injection_task_id): r for r in rows}
    r = by[("user_task_0", "injection_task_0")]
    assert r.selector_applied == 1 and r.selector_original_tokens == 100 and r.selector_delivered_tokens == 40
    assert r.selector_request_ids == [r.case_id + "/t0"]
    assert by[("user_task_0", "injection_task_1")].selector_passthrough == 1
    assert r.tool_call_count == 2 and r.model_call_count == 2 and r.termination == "final_answer"
    assert r.attack_success is False and by[("user_task_1", "injection_task_1")].utility is False


def test_security_true_means_attack_succeeded(tmp_path):
    p = write_result_file(tmp_path, "user_task_0", "injection_task_0", utility=False, security=True)
    from csbench.suites.agentdojo.collect import load_case_row

    row = load_case_row(p, repeat=1, arm=ARM_BASELINE)
    assert row.attack_success is True and row.utility is False


# ---------------------------------------------------------- pair matching


def test_pair_matching_rejects_different_case_sets():
    a = _rows(sorted(expected_case_ids(SUITE, ATTACK, ["user_task_0", "user_task_1"], ["injection_task_0"], 1, ARM_BASELINE)))
    b = _rows(sorted(expected_case_ids(SUITE, ATTACK, ["user_task_0", "user_task_2"], ["injection_task_0"], 1, ARM_NEEDLEPATH)))
    with pytest.raises(PairMismatch):
        validate_pair(a, b)
    validate_pair(a, _rows(sorted(expected_case_ids(SUITE, ATTACK, ["user_task_0", "user_task_1"], ["injection_task_0"], 1, ARM_NEEDLEPATH))))
    assert pair_id(SUITE, ATTACK, "user_task_0", "injection_task_0", 1) == a[0].pair_id


# ----------------------------------------------------------------- runner


def test_runner_happy_path_writes_manifest_and_accepts_every_arm(tmp_path):
    cfg = small_config(repeats=2)
    launcher = FakeLauncher(cfg)
    runner = make_runner(tmp_path, cfg, launcher)
    manifest = runner.run()
    assert len(manifest.accepted) == 4 and all(a.valid for a in manifest.attempts)
    # run order: repeat 1 baseline -> selector, repeat 2 baseline -> selector; two shards per wave
    order = [(r, a) for r, a, _, _ in launcher.calls]
    assert order == [(1, ARM_BASELINE)] * 2 + [(1, ARM_NEEDLEPATH)] * 2 + [(2, ARM_BASELINE)] * 2 + [(2, ARM_NEEDLEPATH)] * 2
    on_disk = RunManifest.load(runner.manifest_path)
    assert on_disk.accepted == manifest.accepted
    att = on_disk.accepted_attempt(1, ARM_NEEDLEPATH)
    assert att.row_count == 4 and att.shard_exit_codes == {"0": 0, "1": 0} and att.worker_count == 2
    arm_dir = cfg.arm_dir(runner.run_root, 1, ARM_NEEDLEPATH)
    assert (arm_dir / "timing.json").exists() and (arm_dir / "console-shard-0.log").exists()
    assert sorted(p.name for p in arm_dir.glob("needlepath_metrics_shard_*.jsonl")) == ["needlepath_metrics_shard_0.jsonl", "needlepath_metrics_shard_1.jsonl"]


def test_runner_retries_a_short_arm_and_archives_the_failed_attempt(tmp_path):
    cfg = small_config(repeats=1)
    launcher = FakeLauncher(cfg, plan={(1, ARM_NEEDLEPATH, 1): "drop_one"})
    runner = make_runner(tmp_path, cfg, launcher)
    manifest = runner.run()
    attempts = [a for a in manifest.attempts if a.arm == ARM_NEEDLEPATH]
    assert [a.valid for a in attempts] == [False, True]
    assert attempts[0].invalidation_reason == "missing_case_ids:1" and attempts[0].row_count == 3
    assert attempts[0].archived_to.startswith(f"{ARM_NEEDLEPATH}.failed_attempt_")
    assert attempts[0].attempt_id.endswith("/attempt_1") and attempts[1].attempt_id.endswith("/attempt_2")
    assert attempts[0].parent_run_id == cfg.run_id
    archived = runner.run_root / "repeat_1" / attempts[0].archived_to
    assert archived.is_dir() and len(list(archived.rglob("*.json"))) >= 3
    assert manifest.accepted[RunManifest.accepted_key(1, ARM_NEEDLEPATH)] == attempts[1].attempt_id


def test_retry_isolation_starts_empty_and_never_reads_the_archive(tmp_path):
    cfg = small_config(repeats=1)
    launcher = FakeLauncher(cfg, plan={(1, ARM_NEEDLEPATH, 1): "exit:1"})
    runner = make_runner(tmp_path, cfg, launcher)
    manifest = runner.run()
    # the retry's shards started in an empty directory (only this shard's own
    # console log may exist when the launcher looks)
    retry_dirs = launcher.seen_dir_contents[-2:]
    assert all(all(n.startswith("console-shard-") for n in names) for names in retry_dirs)
    arm_dir = cfg.arm_dir(runner.run_root, 1, ARM_NEEDLEPATH)
    rows, events = collect_arm(arm_dir, repeat=1, arm=ARM_NEEDLEPATH)
    assert len(rows) == 4 and len(events) == 4, "exactly one attempt's output; nothing appended from the archive"
    failed = [p for p in (runner.run_root / "repeat_1").iterdir() if ".failed_attempt_" in p.name]
    assert len(failed) == 1 and len(list(failed[0].glob("needlepath_metrics_shard_*.jsonl"))) == 2
    assert [a.invalidation_reason for a in manifest.attempts if a.arm == ARM_NEEDLEPATH][0] == "shard_exit_nonzero:['0', '1']"


def test_runner_gives_up_after_three_attempts(tmp_path):
    cfg = small_config(repeats=1)
    plan = {(1, ARM_BASELINE, n): "error_row" for n in (1, 2, 3)}
    launcher = FakeLauncher(cfg, plan=plan)
    runner = make_runner(tmp_path, cfg, launcher)
    with pytest.raises(ArmFailed, match="3 attempt"):
        runner.run()
    manifest = RunManifest.load(runner.manifest_path)
    assert [a.invalidation_reason for a in manifest.attempts] == ["provider_error_rows:2"] * 3
    assert manifest.accepted == {}
    assert len([p for p in (runner.run_root / "repeat_1").iterdir() if ".failed_attempt_" in p.name]) == 3


def test_runner_archives_a_preexisting_directory_instead_of_reusing_it(tmp_path):
    cfg = small_config(repeats=1)
    runner = make_runner(tmp_path, cfg, FakeLauncher(cfg))
    stale = cfg.arm_dir(runner.run_root, 1, ARM_BASELINE)
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("from an interrupted process")
    runner.run()
    assert not (stale / "leftover.txt").exists()
    assert any(".failed_attempt_" in p.name for p in (runner.run_root / "repeat_1").iterdir())


def test_archive_never_overwrites(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    a = archive_failed_attempt(d, "S")
    d.mkdir()
    b = archive_failed_attempt(d, "S")
    assert a != b and a.exists() and b.exists()


def test_manifest_is_validated_before_any_shard_runs(tmp_path):
    cfg = small_config(repeats=1)
    env = fake_environment()
    env.harness_commit = None  # a required environment field
    launcher = FakeLauncher(cfg)
    runner = Runner(cfg, tmp_path / "run", launcher=launcher, environment=env)
    from csbench.suites.agentdojo.config import ManifestError

    with pytest.raises(ManifestError):
        runner.run()
    assert launcher.calls == []


def test_manifest_has_no_secret_values(tmp_path, monkeypatch):
    secret = "fake-selector-key-0123456789abcdef"
    monkeypatch.setenv("NEEDLEPATH_API_KEY", secret)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-model-key-0123456789abcdef")
    cfg = small_config(repeats=1)
    runner = make_runner(tmp_path, cfg, FakeLauncher(cfg))
    runner.run()
    text = runner.manifest_path.read_text() + runner.config_path.read_text()
    assert secret not in text and "fake-model-key" not in text
    assert json.loads(runner.config_path.read_text())["selector"]["api_key_env"] == ["NEEDLEPATH_API_KEY"]
