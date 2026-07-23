import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_scrub_gate_is_clean_on_this_repo():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "scrub_check.py"), "--root", str(REPO)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"scrub gate found hits:\n{result.stdout}\n{result.stderr}"


def test_scrub_gate_catches_a_planted_secret(tmp_path):
    # A throwaway git repo with a planted private-IP literal must fail the gate.
    # The literal is assembled from fragments so this test file itself stays
    # scrub-clean (it must not contain the literal it is testing for).
    planted = "db_host = " + "10." + "1.2." + "3" + "\n"
    (tmp_path / "f.txt").write_text(planted)
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "scrub_check.py"), "--root", str(tmp_path), "--skip-commits"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "infra:private-ip" in result.stdout


def test_scrub_gate_scopes_author_names_to_sanctioned_locations(tmp_path):
    # The paper's byline authors may be named under paper/ (and CITATION.cff /
    # .zenodo.json) but nowhere else. Assemble the name from fragments so THIS
    # test file (not a sanctioned location) stays scrub-clean.
    name = "Sw" + "anand " + "R" + "ao"
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "byline.tex").write_text(name + "\n")          # sanctioned -> allowed
    (tmp_path / "csbench").mkdir()
    (tmp_path / "csbench" / "leak.py").write_text("# " + name + "\n")    # elsewhere -> banned
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "scrub_check.py"), "--root", str(tmp_path), "--skip-commits"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "name:author-outside-sanctioned" in result.stdout
    assert "csbench/leak.py" in result.stdout          # the non-sanctioned file is flagged
    assert "paper/byline.tex" not in result.stdout     # the sanctioned file is not
