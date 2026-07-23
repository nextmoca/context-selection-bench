import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "verify_manifests.py"


def _write_outputs(base: Path) -> str:
    base.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, content in [("a.jsonl", b"row-a\n"), ("b.jsonl", b"row-b\n")]:
        (base / name).write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        lines.append(f"{digest}  {name}")
    return "\n".join(lines) + "\n"


def test_manifest_matches(tmp_path):
    outdir = tmp_path / "out"
    manifest = tmp_path / "m.sha256"
    manifest.write_text(_write_outputs(outdir))
    r = subprocess.run(
        [sys.executable, str(TOOL), str(manifest), "--base-dir", str(outdir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout
    assert "2/2 files verified" in r.stdout


def test_manifest_detects_tamper(tmp_path):
    outdir = tmp_path / "out"
    manifest = tmp_path / "m.sha256"
    manifest.write_text(_write_outputs(outdir))
    (outdir / "a.jsonl").write_bytes(b"tampered\n")  # corrupt one file
    r = subprocess.run(
        [sys.executable, str(TOOL), str(manifest), "--base-dir", str(outdir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "MISMATCH" in r.stdout


def test_manifest_detects_missing(tmp_path):
    outdir = tmp_path / "out"
    manifest = tmp_path / "m.sha256"
    manifest.write_text(_write_outputs(outdir))
    (outdir / "b.jsonl").unlink()  # remove one file
    r = subprocess.run(
        [sys.executable, str(TOOL), str(manifest), "--base-dir", str(outdir)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "MISSING" in r.stdout
