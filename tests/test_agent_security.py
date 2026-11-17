"""Agent-security suites: scorer, pinned configurations and committed results.

Offline. The scorer is exercised on synthetic AgentDojo-format row files; the
committed configurations, summaries and README tables are checked against
each other and against the published headline counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from csbench.suites import agent_security as A

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "agentdojo": ROOT / "csbench" / "suites" / "agentdojo" / "published",
    "agentdyn": ROOT / "csbench" / "suites" / "agentdyn",
    "autodojo": ROOT / "csbench" / "suites" / "autodojo",
}
SUITE_READMES = {
    "agentdojo": ROOT / "csbench" / "suites" / "agentdojo" / "README.md",
    "agentdyn": SUITES["agentdyn"] / "README.md",
    "autodojo": SUITES["autodojo"] / "README.md",
}
TOP_README_NAMES = {"agentdojo": "AgentDojo", "agentdyn": "AgentDyn", "autodojo": "AutoDojo (adaptive)"}

# Published headline counts (pooled, attacked tasks). Presence pins: a change
# to a committed summary must change these on purpose.
PUBLISHED = {
    "agentdojo": {"n": 949, "asr": (268, 0), "uua": (533, 677), "benign": ((76, 97), (71, 97))},
    "agentdyn": {"n": 560, "asr": (217, 35), "uua": (316, 322), "benign": ((36, 60), (37, 60))},
    "autodojo": {"n": 144, "asr": (93, 17), "uua": (118, 116), "benign": ((14, 16), (14, 16))},
}

# Operating-point labels that are listed by the hosted service and were used
# by a run here. Any other label-shaped string in these files is a defect.
LISTED_LABELS_USED = {"np-2026-10-r8"}
_LABEL = re.compile(r"\bnp-\d{4}-\d{2}-[a-z0-9-]+\b")


def config(name):
    return A.load_json(SUITES[name] / "run_config.json")


def summary(name):
    return A.load_json(SUITES[name] / "results" / "summary.json")


# --------------------------------------------------------------- synthetic rows

def write_row(root: Path, pipeline, suite, user_task, attack, injection, utility, security=None, telemetry=None, **extra):
    d = root / pipeline / suite / user_task / (attack or "none")
    d.mkdir(parents=True, exist_ok=True)
    name = f"{injection}.json" if injection else "none.json"
    messages = [{"role": "tool", "content": "x"}]
    if "next_moca" in pipeline:
        messages[0]["next_moca"] = telemetry or {
            "request_id": f"agentdojo/{suite}/{user_task}/{injection or 'none'}/op-a/t0",
            "decision": "applied", "reason": "applied", "operating_point": "op-a", "tool_results": 1, "receipt": {},
        }
    body = {
        "messages": messages,
        "suite_name": suite,
        "pipeline_name": pipeline,
        "user_task_id": user_task,
        "injection_task_id": injection,
        "attack_type": attack,
        "utility": utility,
        "security": security,
        **extra,
    }
    (d / name).write_text(json.dumps(body), encoding="utf-8")


def tiny_config():
    return {
        "schema": A.CONFIG_SCHEMA,
        "suite": "tiny",
        "benchmark": {
            "name": "Tiny",
            "upstream_commit": "0" * 40,
            "attack": "important_instructions",
            "suites": {
                "banking": {
                    "user_tasks": 2,
                    "injection_tasks": 2,
                    "attacked": 4,
                    "benign": 2,
                    "user_task_ids": ["user_task_0", "user_task_1"],
                    "injection_task_ids": ["injection_task_0", "injection_task_1"],
                }
            },
        },
        "fork": {"repo": "https://github.com/nextmoca/tiny", "commit": "1" * 40},
        "model": {"id": "m"},
        "arms": {"none": {"pipeline_name": "m"}, "next_moca": {"build_id": "b", "pipeline_name": "m-next_moca"}},
        "method": ["one line"],
    }


def fill_arm(root: Path, pipeline: str, attack_success: set, utility_fail: set = frozenset(), injections=("injection_task_0", "injection_task_1"), telemetry=None):
    for u in ("user_task_0", "user_task_1"):
        write_row(root, pipeline, "banking", u, None, None, u not in utility_fail, telemetry=telemetry)
        for i in injections:
            write_row(root, pipeline, "banking", u, "important_instructions", i, (u, i) not in utility_fail, (u, i) in attack_success, telemetry=telemetry)
    # AgentDojo's injection-as-user rows: counted, never scored
    write_row(root, pipeline, "banking", "injection_task_0", None, None, True)


def test_scan_classifies_and_counts_rows(tmp_path):
    fill_arm(tmp_path / "none", "m", {("user_task_0", "injection_task_1")}, {("user_task_1", "injection_task_0")})
    scan = A.scan_arm_dir(tmp_path / "none", "none", "important_instructions", ["banking"])
    assert scan.injection_as_user == {"banking": 1}
    assert (scan.incomplete, scan.unreadable, scan.other) == (0, 0, 0)
    counts = A.count_rows(scan.rows, ["banking"])["banking"]
    assert counts == {
        "attacked_n": 4,
        "attack_success_k": 1,
        "utility_under_attack_k": 3,
        "benign_n": 2,
        "benign_utility_k": 2,
        "error_rows": 0,
    }


@pytest.mark.parametrize("field,value", [("security", "yes"), ("security", None), ("utility", 1)])
def test_a_non_boolean_outcome_is_refused(tmp_path, field, value):
    fill_arm(tmp_path / "a", "m", set())
    kwargs = {"utility": True, "security": False, field: value}
    write_row(tmp_path / "a", "m", "banking", "user_task_0", "important_instructions", "injection_task_0", kwargs["utility"], kwargs["security"])
    with pytest.raises(A.ScoreError, match="boolean"):
        A.scan_arm_dir(tmp_path / "a", "none", "important_instructions", ["banking"])


def test_a_row_filed_under_the_wrong_suite_is_refused(tmp_path):
    fill_arm(tmp_path / "a", "m", set())
    f = tmp_path / "a" / "m" / "banking" / "user_task_0" / "important_instructions" / "injection_task_0.json"
    d = json.loads(f.read_text())
    d["suite_name"] = "slack"
    f.write_text(json.dumps(d))
    with pytest.raises(A.ScoreError, match="does not match its directory"):
        A.scan_arm_dir(tmp_path / "a", "none", "important_instructions", ["banking"])


def test_incomplete_and_unreadable_rows_are_set_aside(tmp_path):
    fill_arm(tmp_path / "a", "m", set())
    d = tmp_path / "a" / "m" / "banking" / "user_task_9" / "important_instructions"
    d.mkdir(parents=True)
    (d / "injection_task_0.json").write_text("{not json", encoding="utf-8")
    (d / "injection_task_1.json").write_text(json.dumps({"attack_type": "important_instructions"}), encoding="utf-8")
    scan = A.scan_arm_dir(tmp_path / "a", "none", "important_instructions", ["banking"])
    assert (scan.unreadable, scan.incomplete) == (1, 1)


def test_one_missing_attacked_row_fails_the_arm(tmp_path):
    fill_arm(tmp_path / "a", "m", set())
    os.remove(tmp_path / "a" / "m" / "banking" / "user_task_1" / "important_instructions" / "injection_task_1.json")
    scan = A.scan_arm_dir(tmp_path / "a", "none", "important_instructions", ["banking"])
    with pytest.raises(A.ScoreError, match="attacked_n 3 != expected 4"):
        A.check_expected(A.count_rows(scan.rows, ["banking"]), A.expected_counts(tiny_config()), "none")


def test_a_case_present_twice_is_refused(tmp_path):
    fill_arm(tmp_path / "a", "m", set())
    fill_arm(tmp_path / "a", "m-copy", set())
    with pytest.raises(A.ScoreError, match="duplicate row"):
        A.scan_arm_dir(tmp_path / "a", "none", "important_instructions", ["banking"])


def test_unpaired_arms_are_refused(tmp_path):
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    os.remove(tmp_path / "nm" / "m-next_moca" / "banking" / "user_task_0" / "important_instructions" / "injection_task_0.json")
    a = A.scan_arm_dir(tmp_path / "none", "none", "important_instructions", ["banking"]).rows
    b = A.scan_arm_dir(tmp_path / "nm", "next_moca", "important_instructions", ["banking"]).rows
    with pytest.raises(A.ScoreError, match="not paired"):
        A.check_pairing(a, b)
    with pytest.raises(A.ScoreError, match="exactly the planned"):
        A.summary_from_rows(tiny_config(), a + b)


def test_a_relabelled_injection_column_is_refused():
    # renaming one injection task everywhere keeps every count and the cross
    # product intact; only the exact planned ids catch it
    rows = A.read_rows_jsonl(SUITES["agentdojo"] / "results" / "rows.jsonl")
    rows = [
        A.Row(r.arm, r.suite, r.user_task_id, "injection_task_99", r.attack, r.utility, r.security, r.error)
        if r.suite == "slack" and r.injection_task_id == "injection_task_1"
        else r
        for r in rows
    ]
    with pytest.raises(A.ScoreError, match="slack"):
        A.summary_from_rows(config("agentdojo"), rows)


def test_committed_rows_with_an_error_are_refused():
    rows = A.read_rows_jsonl(SUITES["agentdojo"] / "results" / "rows.jsonl")
    r = rows[0]
    rows[0] = A.Row(r.arm, r.suite, r.user_task_id, r.injection_task_id, r.attack, r.utility, r.security, True)
    with pytest.raises(A.ScoreError, match="provider error"):
        A.summary_from_rows(config("agentdojo"), rows)


def test_score_command_writes_a_summary_and_rows_that_check(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "results").mkdir(parents=True)
    (pkg / "run_config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    fill_arm(tmp_path / "none", "m", {("user_task_0", "injection_task_0"), ("user_task_1", "injection_task_1")})
    fill_arm(tmp_path / "nm", "m-next_moca", set(), {("user_task_0", "injection_task_1")})
    out = pkg / "results" / "summary.json"
    assert A.cli(pkg, "t", ["score", "--none", str(tmp_path / "none"), "--next-moca", str(tmp_path / "nm"), "--out", str(out), "--build-id", "b", "--operating-point", "op-a"]) == 0
    s = A.load_json(out)
    assert s["arms"]["none"]["pooled"]["attack_success_k"] == 2
    assert s["arms"]["next_moca"]["pooled"]["utility_under_attack_k"] == 3
    assert len(A.read_rows_jsonl(pkg / "results" / "rows.jsonl")) == 12
    assert A.cli(pkg, "t", ["check"]) == 0


def _pkg(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "results").mkdir(parents=True)
    (pkg / "run_config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    return pkg


def _score(pkg, none_dir, nm_dir, build_id="b", op="op-a"):
    return A.cli(pkg, "t", ["score", "--none", str(none_dir), "--next-moca", str(nm_dir), "--build-id", build_id, "--operating-point", op])


def test_score_refuses_the_same_directory_for_both_arms(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    with pytest.raises(A.ScoreError, match="not all next_moca pipelines|not the configured"):
        _score(pkg, tmp_path / "none", tmp_path / "none")


def test_score_refuses_a_defended_arm_without_telemetry(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    for f in (tmp_path / "nm").rglob("*.json"):
        d = json.loads(f.read_text())
        d["messages"] = [{"role": "tool", "content": "x"}]
        f.write_text(json.dumps(d))
    with pytest.raises(A.ScoreError, match="no row carries next_moca telemetry"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_score_refuses_rows_with_a_provider_error(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    write_row(tmp_path / "nm", "m-next_moca", "banking", "user_task_1", "important_instructions", "injection_task_1", False, False, error="RateLimitError")
    with pytest.raises(A.ScoreError, match="provider error"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_a_fresh_summary_validates_against_its_own_run_config(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    out = tmp_path / "fresh" / "summary.json"
    A.cli(pkg, "t", ["score", "--none", str(tmp_path / "none"), "--next-moca", str(tmp_path / "nm"), "--out", str(out),
                     "--build-id", "fresh123", "--operating-point", "op-a", "--service-url", "http://svc.example",
                     "--run-started", "2026-11-17T00:00:00Z", "--run-ended", "2026-11-17T01:00:00Z"])
    s, cfg = A.load_json(out), A.load_json(out.with_name("run_config.json"))
    assert all("fresh123" in line or "op-a" not in line for line in cfg["method"])
    assert any("http://svc.example, build fresh123, operating point op-a" in line for line in cfg["method"])
    assert cfg["arms"]["next_moca"]["service_url"] == "http://svc.example"
    assert cfg["run"]["started_utc"] == "2026-11-17T00:00:00Z" and cfg["run"]["ended_utc"] == "2026-11-17T01:00:00Z"
    assert s["build_id"] == cfg["arms"]["next_moca"]["build_id"] == "fresh123"
    assert cfg["arms"]["next_moca"]["operating_point"] == "op-a"
    assert s["next_moca_telemetry"]["operating_points"] == ["op-a"]
    A.validate_summary(s, cfg)
    with pytest.raises(A.ScoreError, match="build_id"):
        A.validate_summary(s, tiny_config())


def test_score_refuses_telemetry_from_another_operating_point(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    with pytest.raises(A.ScoreError, match="operating points"):
        A.cli(pkg, "t", ["score", "--none", str(tmp_path / "none"), "--next-moca", str(tmp_path / "nm"),
                         "--build-id", "b", "--operating-point", "op-b"])


def test_score_refuses_a_run_that_never_reached_the_service(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    for u in ("user_task_0", "user_task_1"):
        for i in (None, "injection_task_0", "injection_task_1"):
            failed = {"request_id": f"agentdojo/banking/{u}/{i or 'none'}/op-a/t0", "decision": "passthrough",
                      "reason": "exception:ConnectionError", "operating_point": "op-a", "tool_results": 1, "receipt": {}}
            write_row(tmp_path / "nm", "m-next_moca", "banking", u, "important_instructions" if i else None, i,
                      True, False if i else None, telemetry=failed)
    with pytest.raises(A.ScoreError, match="never reached"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_score_refuses_an_empty_build_id(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    with pytest.raises(A.ScoreError, match="build-id"):
        _score(pkg, tmp_path / "none", tmp_path / "nm", build_id=" ")


def test_a_baseline_pipeline_named_for_the_defense_is_refused(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m/next_moca_off", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set())
    with pytest.raises(A.ScoreError, match="not the configured"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_a_different_model_in_the_defended_arm_is_refused(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "other-model-next_moca", set())
    with pytest.raises(A.ScoreError, match="not the configured"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


_GOOD = {"request_id": "agentdojo/banking/user_task_0/none/op-a/t0", "decision": "applied", "reason": "applied",
         "operating_point": "op-a", "tool_results": 1, "receipt": {}}


@pytest.mark.parametrize("record", [
    {k: v for k, v in _GOOD.items() if k != "request_id"},
    {**_GOOD, "decision": "kept"},
    {k: v for k, v in _GOOD.items() if k != "operating_point"},
    {**_GOOD, "reason": ""},
    {**_GOOD, "tool_results": 0},
    {k: v for k, v in _GOOD.items() if k != "receipt"},
    {},
])
def test_malformed_telemetry_is_refused(tmp_path, record):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set(), telemetry=record or {"x": 1})
    with pytest.raises(A.ScoreError, match="malformed next_moca telemetry"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_telemetry_from_another_row_or_reused_is_refused(tmp_path):
    pkg = _pkg(tmp_path)
    fill_arm(tmp_path / "none", "m", set())
    fill_arm(tmp_path / "nm", "m-next_moca", set(), telemetry=_GOOD)
    with pytest.raises(A.ScoreError, match="does not belong to this row|already seen"):
        _score(pkg, tmp_path / "none", tmp_path / "nm")


def test_score_requires_the_operating_point():
    with pytest.raises(SystemExit):
        A.cli(SUITES["autodojo"], "t", ["score", "--none", "a", "--next-moca", "b", "--build-id", "x"])


def test_duplicated_jsonl_rows_are_refused():
    rows = A.read_rows_jsonl(SUITES["agentdojo"] / "results" / "rows.jsonl")
    # replace one attacked case with a copy of another in both arms: totals unchanged
    for arm in A.ARMS:
        idx = [i for i, r in enumerate(rows) if r.arm == arm and r.kind == "attacked"]
        rows[idx[1]] = rows[idx[0]]
    with pytest.raises(A.ScoreError, match="duplicated cases"):
        A.summary_from_rows(config("agentdojo"), rows)


def test_a_bogus_task_replacing_a_real_pair_is_refused():
    rows = A.read_rows_jsonl(SUITES["agentdojo"] / "results" / "rows.jsonl")
    for arm in A.ARMS:
        i = next(i for i, r in enumerate(rows) if r.arm == arm and r.kind == "attacked" and r.suite == "slack")
        r = rows[i]
        rows[i] = A.Row(r.arm, r.suite, r.user_task_id, "injection_task_99", r.attack, r.utility, r.security, r.error)
    with pytest.raises(A.ScoreError, match="slack"):
        A.summary_from_rows(config("agentdojo"), rows)


def test_percentages_round_half_up():
    # 35/560 = 6.25 and 217/560 = 38.75: banker's rounding would print 6.2 and 38.8
    assert A.pct1(35, 560) == "6.3"
    assert A.pct1(217, 560) == "38.8"
    assert A.pct(1, 8) == 12.5 and A.pct(1, 3) == 33.33
    assert A.pct(0, 0) is None


def test_pin_value_formats_for_the_shell():
    cfg = tiny_config()
    assert A.pin_value(cfg, "fork.commit") == "1" * 40
    assert A.pin_value(cfg, "benchmark.suites") == "banking"
    assert A.pin_value({"a": None, "b": True, "c": [1, 2]}, "a") == ""
    assert A.pin_value({"a": None, "b": True, "c": [1, 2]}, "b") == "1"
    assert A.pin_value({"a": None, "b": True, "c": [1, 2]}, "c") == "1 2"
    with pytest.raises(A.ScoreError):
        A.pin_value(cfg, "fork.branch")


# --------------------------------------------------------------- configurations

@pytest.mark.parametrize("name", sorted(SUITES))
def test_committed_configuration_is_valid(name):
    A.validate_config(config(name))


@pytest.mark.parametrize("bad", ["next-moca", "f9010f8", "F" * 40])
def test_a_branch_name_or_short_sha_is_not_a_pin(bad):
    cfg = tiny_config()
    cfg["fork"]["commit"] = bad
    with pytest.raises(A.ScoreError, match="40-hex"):
        A.validate_config(cfg)


def test_a_configuration_without_a_build_id_is_refused():
    cfg = tiny_config()
    cfg["arms"]["next_moca"] = {}
    with pytest.raises(A.ScoreError, match="build_id"):
        A.validate_config(cfg)


def test_population_must_be_the_full_cross_product():
    cfg = tiny_config()
    cfg["benchmark"]["suites"]["banking"]["attacked"] = 3
    with pytest.raises(A.ScoreError, match="user_tasks x injection_tasks"):
        A.validate_config(cfg)


@pytest.mark.parametrize("name", sorted(SUITES))
def test_every_fork_is_pinned_to_its_next_moca_branch_commit(name):
    fork = config(name)["fork"]
    assert fork["branch"] == "next-moca"
    assert fork["repo"] == f"https://github.com/nextmoca/{name}"
    assert re.fullmatch(r"[0-9a-f]{40}", fork["commit"])
    assert fork["element"]["sha256_16"] == "b9698486c20eb54d"


# --------------------------------------------------------------- committed results

@pytest.mark.parametrize("name", sorted(SUITES))
def test_committed_summary_follows_from_its_counts(name):
    A.validate_summary(summary(name), config(name))


def test_a_summary_with_an_edited_rate_is_refused():
    s = copy.deepcopy(summary("agentdyn"))
    s["arms"]["next_moca"]["pooled"]["attack_success_pct"] = 5.0
    with pytest.raises(A.ScoreError, match="do not follow"):
        A.validate_summary(s, config("agentdyn"))


def test_a_summary_pinned_to_another_commit_is_refused():
    s = copy.deepcopy(summary("autodojo"))
    s["fork_commit"] = "2" * 40
    with pytest.raises(A.ScoreError, match="fork_commit"):
        A.validate_summary(s, config("autodojo"))


@pytest.mark.parametrize("key,value", [("model", "gpt-4o"), ("build_id", "0" * 16), ("attack", "direct"), ("benchmark", "AgentDojo")])
def test_a_summary_with_an_edited_header_is_refused(key, value):
    s = copy.deepcopy(summary("agentdyn"))
    s[key] = value
    with pytest.raises(A.ScoreError, match=key):
        A.validate_summary(s, config("agentdyn"))


def test_a_count_above_its_population_is_refused():
    s = copy.deepcopy(summary("autodojo"))
    c = s["arms"]["none"]["per_suite"]["banking"]
    c["benign_utility_k"] = 17
    with pytest.raises(A.ScoreError, match="exceeds its population"):
        A.validate_summary(s, config("autodojo"))


def test_a_summary_with_an_extra_arm_is_refused():
    s = copy.deepcopy(summary("autodojo"))
    s["arms"]["other"] = s["arms"]["none"]
    with pytest.raises(A.ScoreError, match="summary arms"):
        A.validate_summary(s, config("autodojo"))


@pytest.mark.parametrize("name", sorted(SUITES))
def test_published_headline_counts(name):
    pub = PUBLISHED[name]
    s = summary(name)
    a, b = s["arms"]["none"]["pooled"], s["arms"]["next_moca"]["pooled"]
    assert a["attacked_n"] == b["attacked_n"] == pub["n"]
    assert (a["attack_success_k"], b["attack_success_k"]) == pub["asr"]
    assert (a["utility_under_attack_k"], b["utility_under_attack_k"]) == pub["uua"]
    assert ((a["benign_utility_k"], a["benign_n"]), (b["benign_utility_k"], b["benign_n"])) == pub["benign"]
    assert a["error_rows"] == b["error_rows"] == 0
    assert s["build_id"] == config(name)["arms"]["next_moca"]["build_id"]


# Rows per arm (attacked + benign) in each committed rows.jsonl.
PUBLISHED_ROWS_PER_ARM = {"agentdojo": 949 + 97, "agentdyn": 560 + 60, "autodojo": 144 + 16}


@pytest.mark.parametrize("name", sorted(SUITES))
def test_committed_rows_reproduce_the_summary(name):
    rows = A.read_rows_jsonl(SUITES[name] / "results" / "rows.jsonl")
    assert len(rows) == 2 * PUBLISHED_ROWS_PER_ARM[name]
    s = summary(name)
    assert s["source"] == "rows"
    rebuilt = A.summary_from_rows(config(name), rows)
    assert rebuilt["arms"] == s["arms"]


# sha256 of each committed rows.jsonl. Swapping two outcomes inside one arm and
# suite keeps every count; only the bytes catch it.
PUBLISHED_ROWS_SHA256 = {
    "agentdojo": "ad8bc11d0fe581fb38e699bf282f3ffa8ae7d0683403b3194331c7ce71ad5540",
    "agentdyn": "50ff72be871a81dfea1a6b7ee9d83f1df8704ea0265f4d37322db73ffac9c229",
    "autodojo": "21fc3870ea4de17559fbbdf9182a205f46621416cb7ea989a9a5f5de616a9307",
}


@pytest.mark.parametrize("name", sorted(SUITES))
def test_committed_rows_are_the_published_bytes(name):
    data = (SUITES[name] / "results" / "rows.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == PUBLISHED_ROWS_SHA256[name]


def test_check_refuses_a_row_sourced_summary_without_its_rows(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "results").mkdir(parents=True)
    src = SUITES["autodojo"]
    (pkg / "run_config.json").write_bytes((src / "run_config.json").read_bytes())
    (pkg / "results" / "summary.json").write_bytes((src / "results" / "summary.json").read_bytes())
    with pytest.raises(A.ScoreError, match="rows.jsonl is missing"):
        A.cli(pkg, "t", ["check"])


def test_rows_naming_another_attack_are_refused():
    # relabelling every attacked row in both arms keeps every count and the pairing
    rows = A.read_rows_jsonl(SUITES["autodojo"] / "results" / "rows.jsonl")
    rows = [
        A.Row(r.arm, r.suite, r.user_task_id, r.injection_task_id, "other_attack", r.utility, r.security, r.error)
        if r.attack is not None
        else r
        for r in rows
    ]
    with pytest.raises(A.ScoreError, match="configured attack"):
        A.summary_from_rows(config("autodojo"), rows)


@pytest.mark.parametrize("name", sorted(SUITES))
def test_check_command_passes(name, capsys):
    assert A.cli(SUITES[name], "t", ["check"]) == 0
    assert "| pooled |" in capsys.readouterr().out or name == "autodojo"


# --------------------------------------------------------------- documentation

@pytest.mark.parametrize("name", sorted(SUITES))
def test_suite_readme_table_is_generated_from_the_summary(name):
    assert A.per_suite_table(summary(name)) in SUITE_READMES[name].read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(SUITES))
def test_top_readme_row_is_generated_from_the_summary(name):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert A.table_row(TOP_README_NAMES[name], summary(name)) in readme


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def _agent_security_texts():
    texts = [(ROOT / "csbench" / "suites" / "agent_security.py").read_text(encoding="utf-8")]
    for pkg in SUITES.values():
        texts += [p.read_text(encoding="utf-8") for p in pkg.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".sh", ".py"}]
    texts.append(_section(ROOT / "README.md", "## Agent security suites"))
    texts.append(_section(SUITE_READMES["agentdojo"], "## Published GPT-4o result"))
    return texts


def test_label_detector_sees_a_label_known_to_be_present():
    text = (SUITES["autodojo"] / "run_config.json").read_text(encoding="utf-8")
    assert _LABEL.findall(text) == ["np-2026-10-r8", "np-2026-10-r8"]


def test_only_listed_operating_point_labels_appear():
    found = set()
    for text in _agent_security_texts():
        found |= set(_LABEL.findall(text))
    assert found, "the detector found no label at all; it is not looking at the right files"
    assert found <= LISTED_LABELS_USED, found - LISTED_LABELS_USED
    assert not any("candidate" in label for label in found)


def test_agentdyn_publishes_a_build_id_and_no_label():
    np_arm = config("agentdyn")["arms"]["next_moca"]
    assert np_arm["operating_point"] is None
    assert re.fullmatch(r"[0-9a-f]{16}", np_arm["build_id"])


# --------------------------------------------------------------- reproduction scripts

@pytest.mark.parametrize("name", sorted(SUITES))
def test_reproduce_script_refuses_without_explicit_opt_in(name):
    env = {k: v for k, v in os.environ.items() if k != "CSBENCH_CONFIRM_PAID_RUN"}
    env["OPENAI_API_KEY"] = env["NEXT_MOCA_API_KEY"] = env["GEMINI_API_KEY"] = "unused"
    proc = subprocess.run(["bash", str(SUITES[name] / "reproduce.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2, proc.stderr
    assert "CSBENCH_CONFIRM_PAID_RUN" in proc.stderr


@pytest.mark.parametrize("name", sorted(SUITES))
def test_reproduce_script_checks_out_the_pinned_commit(name):
    text = (SUITES[name] / "reproduce.sh").read_text(encoding="utf-8")
    assert "pin fork.commit" in text and 'checkout --quiet --detach "$COMMIT"' in text
    assert 'rev-parse HEAD)" = "$COMMIT"' in text
