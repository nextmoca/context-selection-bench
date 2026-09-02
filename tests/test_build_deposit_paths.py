"""Arm directories under items/<length>/ are data, not engine scratch (needlepath arm regression)."""

from pathlib import Path

from tools import build_deposit as bd


def test_needlepath_arm_rows_are_not_excluded():
    assert not bd._excluded_path(Path("items/8k/needlepath/qa_1.jsonl"))
    assert not bd._excluded_path(Path("items/16k/needlepath/cwe.jsonl"))
    assert not bd._excluded_path(Path("items/8k/full_context/vt.jsonl"))


def test_engine_scratch_directories_stay_excluded():
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
                          cwd=Path(bd.__file__).resolve().parents[1])
    assert proc.returncode != 0
    assert "zero per-item records" in (proc.stdout + proc.stderr)
