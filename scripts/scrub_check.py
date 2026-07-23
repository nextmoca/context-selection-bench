#!/usr/bin/env python3
"""Assert-zero scrub gate for the public tree and commit history.

Scans every git-tracked file's contents and every commit message for a set of
universally-innocuous safety patterns that must never appear in a public
repository: credential-shaped literals, private IP addresses, absolute home
paths, a generalized dataset-canary shape, and the paper's byline author names
outside their sanctioned locations.

Exit code 0 = clean, 1 = one or more hits. Wired into CI. This script names the
patterns and is therefore excluded from the tracked-file scan (it would match
itself); it is short and reviewed by hand.

Run:  python scripts/scrub_check.py [--root .]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (name, compiled regex). Case-insensitive unless the pattern is inherently cased.
#
# This public gate holds only UNIVERSALLY-INNOCUOUS protections: secrets, private
# IPs, absolute home paths, a generalized dataset-canary shape, and (separately,
# below) the byline author names outside sanctioned locations. It deliberately
# does not enumerate any project-specific name, because a public pattern that
# spelled one out would itself disclose it.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # --- host / filesystem leakage ---
    ("path:home", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")),
    # --- generic cloud infra / private networks ---
    ("infra:cloud", re.compile(r"\bus-(?:east|west)-\d\b|\bec2-[\d-]+\.compute|\.compute\.amazonaws\.com|\.internal\b")),
    ("infra:private-ip", re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    # --- generalized dataset-canary shape (the exact canary value is not named) ---
    ("canary:shape", re.compile(r"\bappworld:[0-9a-f]{6,}\b", re.IGNORECASE)),
    # --- personal names are handled separately, with a scoped exception for the
    #     paper's byline authors; see _NAME_* and the scoped checks below ---
    # --- credential-shaped literals (provider-neutral) ---
    ("secret:pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("secret:token-sk", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("secret:aws-akid", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret:google", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    ("secret:api-key-literal", re.compile(r"[A-Z][A-Z0-9_]*_API_KEY\s*[=:]\s*['\"][^'\"]+['\"]")),
]

# Files that legitimately contain the pattern literals (they define the gate).
_EXCLUDE = {"scripts/scrub_check.py"}

# Author names (the released paper's byline authors) are PERMITTED only in
# sanctioned authorship locations and an explicit whitelist of authorship
# commits; everywhere else in the tree and history they remain BANNED. This
# keeps the neutral-references discipline while allowing the byline, CRediT,
# conflict-of-interest statement, and citation metadata to name the authors.
_NAME_PATTERN = re.compile(r"\b(?:kiran|kashalkar|swanand|rao)\b", re.IGNORECASE)
_NAME_ALLOWED_PATH_PREFIXES = ("paper/",)
_NAME_ALLOWED_FILES = {"CITATION.cff", ".zenodo.json"}
_NAME_ALLOWED_COMMITS = {
    "9b6e3314f01574ff18608def89f49e79cf8a4a19",  # paper: two-author front matter (byline/CRediT/COI)
    "79019d68bb07d16cc721c84abe52bab9d5a6f13e",  # paper: two-author byline
}


def _name_sanctioned_path(rel: str) -> bool:
    return rel in _NAME_ALLOWED_FILES or any(
        rel.startswith(p) for p in _NAME_ALLOWED_PATH_PREFIXES
    )


def _tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def scan_tree(root: Path) -> list[tuple[str, int, str, str]]:
    hits: list[tuple[str, int, str, str]] = []
    for rel in _tracked_files(root):
        if rel in _EXCLUDE:
            continue
        path = root / rel
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if _looks_binary(raw):
            continue
        text = raw.decode("utf-8", errors="replace")
        name_ok = _name_sanctioned_path(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pat in _PATTERNS:
                if pat.search(line):
                    hits.append((rel, lineno, name, line.strip()[:160]))
            if not name_ok and _NAME_PATTERN.search(line):
                hits.append((rel, lineno, "name:author-outside-sanctioned", line.strip()[:160]))
    return hits


def scan_commits(root: Path) -> list[tuple[str, str, str]]:
    out = subprocess.run(
        ["git", "-C", str(root), "log", "--format=%H%x1f%B%x1e"],
        capture_output=True, text=True, check=True,
    ).stdout
    hits: list[tuple[str, str, str]] = []
    for record in out.split("\x1e"):
        record = record.strip()
        if not record or "\x1f" not in record:
            continue
        sha, body = record.split("\x1f", 1)
        for name, pat in _PATTERNS:
            m = pat.search(body)
            if m:
                hits.append((sha[:10], name, m.group(0)))
        if sha not in _NAME_ALLOWED_COMMITS:
            m = _NAME_PATTERN.search(body)
            if m:
                hits.append((sha[:10], "name:author-outside-sanctioned", m.group(0)))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Assert-zero scrub gate.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-commits", action="store_true", help="scan the tree only")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    tree_hits = scan_tree(root)
    commit_hits = [] if args.skip_commits else scan_commits(root)

    for rel, lineno, name, snippet in tree_hits:
        print(f"TREE   {rel}:{lineno}  [{name}]  {snippet}")
    for sha, name, match in commit_hits:
        print(f"COMMIT {sha}  [{name}]  {match!r}")

    total = len(tree_hits) + len(commit_hits)
    if total:
        print(f"\nscrub gate FAILED: {total} hit(s) ({len(tree_hits)} tree, {len(commit_hits)} commit)")
        return 1
    print("scrub gate clean: 0 hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
