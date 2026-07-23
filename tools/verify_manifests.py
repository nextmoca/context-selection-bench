#!/usr/bin/env python3
"""Verify published per-item benchmark outputs against a committed sha256 manifest.

A manifest is a standard ``sha256sum``-format file: one ``<hexdigest>  <path>``
line per output file. Reproduction workflow for an outsider:

  1. Download the published per-item outputs into a local directory.
  2. Run:  python tools/verify_manifests.py <manifest> --base-dir <dir>
  3. Every file's recomputed sha256 must match the manifest.

Exit code 0 = all match, 1 = any mismatch/missing. Dependency-free.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parse_manifest(text: str) -> list[tuple[str, str]]:
    """Parse ``<hexdigest>  <path>`` lines (sha256sum format)."""
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"malformed manifest line: {line!r}")
        digest, path = parts[0].strip(), parts[1].strip()
        if path.startswith("*"):  # sha256sum binary marker
            path = path[1:]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"not a sha256 hexdigest: {digest!r}")
        entries.append((digest.lower(), path))
    return entries


def resolve(path: str, base_dir: Path, strip_prefix: str | None) -> Path:
    if strip_prefix and path.startswith(strip_prefix):
        path = path[len(strip_prefix):].lstrip("/")
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    return base_dir / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify per-item outputs against a sha256 manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--base-dir", default=".", help="directory the manifest paths resolve against")
    parser.add_argument("--strip-prefix", default=None, help="prefix to strip from each manifest path (e.g. a storage URL)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    entries = parse_manifest(Path(args.manifest).read_text(encoding="utf-8"))

    ok = 0
    failures: list[str] = []
    for expected, rel in entries:
        target = resolve(rel, base_dir, args.strip_prefix)
        if not target.exists():
            failures.append(f"MISSING  {rel}")
            continue
        actual = sha256_file(target)
        if actual == expected:
            ok += 1
        else:
            failures.append(f"MISMATCH {rel}\n           expected {expected}\n           actual   {actual}")

    for f in failures:
        print(f)
    print(f"\n{ok}/{len(entries)} files verified", end="")
    if failures:
        print(f"; {len(failures)} problem(s)")
        return 1
    print(": manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
