"""Reproducibility provenance: JSON-safety, file hashing, git state, manifests.

This module is the harness's provenance surface. It is deliberately pure
(standard library only, no engine, no model, no arm internals) so a skeptical
reader can audit exactly how a published run is pinned and verified:

- ``json_safe`` coerces arbitrary run payloads into JSON-serializable values so
  a row/summary/manifest always round-trips through ``json.dumps``.
- ``file_sha256`` / ``count_lines`` are the two file measures every manifest is
  built from.
- ``probe_git_state`` records a neutral snapshot of the repository the run was
  produced from (commit, branch, dirty flag, and a hash of the working diff),
  so a result file states which tree it came from.
- ``build_trusted_benchmark_manifest`` records the *inputs* of a run: the
  sha256 and line-count of each dataset file, plus the git snapshot and the
  leakage controls the protocol commits to.
- ``build_output_manifest`` / ``write_output_manifest`` record the *outputs* of
  a run: a standard ``sha256sum``-format manifest over the committed per-item
  output files, so anyone can recompute it with ``tools/verify_manifests.py``.

There is no enforce-clean-worktree gate wired to any internal path here; the
optional ``enforce_clean_worktree`` helper is a generic guard a caller may
choose to invoke, nothing more.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #


def json_safe(value: Any) -> Any:
    """Recursively coerce ``value`` into JSON-serializable primitives.

    Dicts and lists/tuples are walked; ``str``/``int``/``float``/``bool``/
    ``None`` pass through unchanged; anything else is stringified via ``str``.
    This lets a run write out arbitrary row/summary payloads without a custom
    JSON encoder.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# File measures
# --------------------------------------------------------------------------- #


def file_sha256(path: Path) -> str:
    """Streaming sha256 hexdigest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    """Count the newline-delimited lines in a file (byte-level, no decode)."""
    with Path(path).open("rb") as handle:
        return sum(1 for _line in handle)


# --------------------------------------------------------------------------- #
# Git state (neutral)
# --------------------------------------------------------------------------- #


def probe_git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Return a neutral snapshot of the git repository at ``repo_root``.

    Records the current commit, branch, a dirty flag, the short status, and a
    sha256 of the working diff (so a dirty state is still pinned to a hash
    rather than dumped verbatim). Any failure (not a git repo, git missing) is
    swallowed into an ``error`` field with an empty snapshot: provenance is
    best-effort and never aborts a run.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()

    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current")
        status_short = run_git("status", "--short")
        diff_text = run_git("diff", "--binary")
    except Exception as exc:
        return {
            "commit": "",
            "branch": "",
            "dirty": True,
            "status_short": "",
            "diff_sha256": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_short),
        "status_short": status_short,
        "diff_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest() if diff_text else "",
    }


def enforce_clean_worktree(git_probe=probe_git_state) -> None:
    """Optional guard: raise if the working tree is dirty.

    Generic and off by default: a caller opts in when it wants a run pinned to
    a committed tree. Not tied to any particular path or run type.
    """
    git_state = git_probe()
    if git_state.get("dirty"):
        raise RuntimeError(
            "Refusing to produce a pinned benchmark run from a dirty git worktree. "
            "Commit or stash changes first, or run without the clean-worktree check."
        )


# --------------------------------------------------------------------------- #
# Trusted-input manifest (dataset provenance)
# --------------------------------------------------------------------------- #


def build_trusted_benchmark_manifest(
    *,
    context_roots: Mapping[str, Path],
    tasks: Sequence[str],
    run_name: str,
    command: str,
    git_probe=probe_git_state,
) -> dict[str, Any]:
    """Record the inputs a run was produced from.

    For every ``(label, root)`` in ``context_roots`` and every task, hash the
    ``<root>/<task>/test.jsonl`` dataset file (sha256 + line count), alongside a
    git snapshot, the Python version, the invoking command, and the leakage
    controls the protocol commits to. This is the *input* provenance record;
    per-item output hashing is done separately by ``build_output_manifest``.
    """
    datasets: dict[str, Any] = {}
    for label, root in context_roots.items():
        task_payload: dict[str, Any] = {}
        for task in tasks:
            path = Path(root) / task / "test.jsonl"
            task_payload[task] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else "",
                "line_count": count_lines(path) if path.exists() else 0,
            }
        datasets[str(label)] = {
            "root": str(root),
            "tasks": task_payload,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "command": command,
        "python": sys.version,
        "git": git_probe(),
        "datasets": datasets,
        "leakage_controls": {
            "expected_answers_excluded_from_selection_keywords": True,
            "all_arms_share_examples": True,
            "all_arms_share_model": True,
            "temperature": 0,
        },
        "benchmark_scope": {
            "suite": "RULER v1 local adapter",
            "official_leaderboard_submission": False,
            "note": (
                "Provenance for a local RULER adapter run; not an official "
                "leaderboard submission."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Output manifest (per-item output provenance, sha256sum format)
# --------------------------------------------------------------------------- #


def build_output_manifest(files: Iterable[Path], *, base_dir: Path) -> str:
    """Build a standard ``sha256sum``-format manifest over per-item outputs.

    Each line is ``<hexdigest>  <relpath>`` where ``relpath`` is the file's
    path relative to ``base_dir`` (POSIX separators), sorted for determinism.
    The output is byte-for-byte consumable by ``tools/verify_manifests.py``
    with ``--base-dir <base_dir>``.
    """
    base = Path(base_dir)
    entries: list[tuple[str, str]] = []
    for path in files:
        p = Path(path)
        rel = os.path.relpath(p, base)
        entries.append((Path(rel).as_posix(), file_sha256(p)))
    entries.sort(key=lambda item: item[0])
    lines = [f"{digest}  {rel}" for rel, digest in entries]
    return "\n".join(lines) + ("\n" if lines else "")


def write_output_manifest(manifest_path: Path, files: Iterable[Path], *, base_dir: Path) -> Path:
    """Write the sha256sum-format output manifest to ``manifest_path``."""
    text = build_output_manifest(files, base_dir=base_dir)
    out = Path(manifest_path)
    out.write_text(text, encoding="utf-8")
    return out
