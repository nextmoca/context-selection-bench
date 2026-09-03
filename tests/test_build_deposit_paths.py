"""Arm directories under items/<length>/ are data, not engine scratch (needlepath arm regression)."""

import importlib.util
from pathlib import Path

# tools/ is a script directory, not a package: load the builder by path so the
# suite runs the way CI runs it (plain `pytest -q` from the repository root).
_SPEC = importlib.util.spec_from_file_location(
    "build_deposit", Path(__file__).resolve().parents[1] / "tools" / "build_deposit.py")
bd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bd)


def test_needlepath_arm_rows_are_not_excluded():
    assert not bd._excluded_path(Path("items/8k/needlepath/qa_1.jsonl"))
    assert not bd._excluded_path(Path("items/16k/needlepath/cwe.jsonl"))
    assert not bd._excluded_path(Path("items/8k/full_context/vt.jsonl"))


def test_engine_scratch_directories_stay_excluded():
    assert bd._excluded_path(Path("items/8k/signals/x.jsonl"))
    assert bd._excluded_path(Path("items/8k/_tmp_run/x.jsonl"))
    assert bd._excluded_path(Path("items/8k/task_dump/x.jsonl"))
    assert bd._excluded_path(Path("items/8k/acon/x.jsonl"))
    assert bd._excluded_path(Path("needlepath/signals/foo.json"))
    assert bd._excluded_path(Path("items/8k/needlepath/signals/foo.json"))
    assert bd._excluded_path(Path("_tmp_run/x.jsonl"))
    assert bd._excluded_path(Path("task_dump/x.json"))
    assert bd._excluded_path(Path("items/8k/compresr/trajectory.jsonl"))


def test_build_ruler_ships_every_arm(tmp_path: Path):
    src = tmp_path / "src"
    for arm in ("full_context", "needlepath", "compresr"):
        d = src / "items" / "8k" / arm
        d.mkdir(parents=True)
        (d / "qa_1.jsonl").write_text('{"arm": "%s", "task": "qa_1", "score": 1.0, "answer": "x"}\n' % arm)
    out = tmp_path / "out"
    out.mkdir()
    n = bd.build_ruler(src, out)
    assert n == 3
    assert sorted(p.name for p in (out / "items" / "8k").iterdir()) == ["compresr", "full_context", "needlepath"]


def test_zero_records_is_a_failure_not_a_deposit(tmp_path: Path, monkeypatch):
    import subprocess, sys
    src = tmp_path / "missing"
    out = tmp_path / "deposit"
    out.mkdir()
    proc = subprocess.run([sys.executable, "tools/build_deposit.py", "--run-type", "ruler", "--run-id", "x",
                           "--src", str(src), "--out", str(out)], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1])
    assert proc.returncode != 0
    assert "nothing to deposit" in (proc.stdout + proc.stderr)


def test_verification_hashes_survive_the_allowlist():
    rec = {"arm": "needlepath", "task": "cwe", "score": 0.5, "item_sha256": "a" * 64,
           "expected_answer_sha256": "b" * 64, "prompt_sha256": "c" * 64, "prompt": "never"}
    out = bd.clean_ruler_record(rec)
    assert out["item_sha256"] == "a" * 64 and out["prompt_sha256"] == "c" * 64
    assert "prompt" not in out


import pytest


@pytest.mark.parametrize("bad", ["RAW PROMPT TEXT", "A" * 64, "ab" * 31, 12345, "0" * 65, None, "0" * 64 + "\n"])
def test_digest_fields_must_be_lowercase_sha256_hex(bad):
    rec = {"arm": "needlepath", "task": "cwe", "score": 0.5, "prompt_sha256": bad}
    with pytest.raises(SystemExit):
        bd.clean_ruler_record(rec)


def test_qa_answers_are_still_hashed_and_digests_kept():
    rec = {"arm": "needlepath", "task": "qa_1", "score": 1.0, "answer": "Paris", "expected_answer": ["Paris"],
           "item_sha256": "0" * 64, "input_sha256": "f" * 64}
    out = bd.clean_ruler_record(rec)
    assert isinstance(out["answer"], dict) and "sha256" in out["answer"]
    assert out["item_sha256"] == "0" * 64 and out["input_sha256"] == "f" * 64
