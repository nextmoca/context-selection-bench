"""Structural assertion that every guard is reachable from a production path.

Standing program rule. This class of failure has now occurred twice: a metric
that was defined but never wired, and (in the very release that fixed it) two
item-identity guards that existed only in tests. A guard that is never called
provides no protection while reading as though it does, and nothing in CI
notices, because its unit tests pass either way.

The instrument is validated before it is trusted: `test_detector_*` below run
the detector against a guard known to be wired and a synthetic guard known not
to be, and assert it gets both right. A detector that cannot fail is not
evidence.

WHAT THIS DETECTOR CANNOT SEE, stated so nobody reads a green run as more than
it is. It matches call sites by name in the AST. It therefore does NOT
establish that a guard actually executes:

  - a call inside `if False:`, inside an uncalled function, or on a branch that
    never runs counts as reachable
  - a call to an unrelated method that happens to share the guard's name counts
    as reachable
  - a guard invoked through `getattr`, an alias, or indirect dispatch is NOT
    seen, and would be reported unreachable

`DEAD_MODULES` covers the one case we know of where a module is on no live path.
This is a syntactic backstop against the specific failure that has now happened
twice -- a guard wired to nothing at all -- not a proof of runtime coverage. A
guard whose execution matters should also have a behavioural test that fails
when it is removed.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "csbench"
GUARD_PREFIXES = ("assert_", "forbid_", "validate_", "verify_", "require_", "ensure_")
GUARD_SUBSTRINGS = ("guard", "tripwire")

# Guards deliberately exempt from the reachability requirement, each with a
# reason. Adding a name here is a decision that should be argued in review, not
# a way to silence the test.
EXEMPT: dict[str, str] = {}

# Modules that are not on any live execution path. A call site inside one of
# these does NOT make a guard reachable: the guard would still never run.
# `ruler/stats.py` reads a rows.json layout the current harness does not emit
# (see docs/audit, finding 8), so it is dead until reconciled or removed.
DEAD_MODULES: tuple[str, ...] = ("csbench/suites/ruler/stats.py",)


def _is_guard(name: str) -> bool:
    return name.startswith(GUARD_PREFIXES) or any(s in name for s in GUARD_SUBSTRINGS)


def _sources() -> list[pathlib.Path]:
    return [p for p in PACKAGE.rglob("*.py") if "vendor" not in p.parts]


def _definitions_and_call_sites():
    """Return ({guard_name: def_location}, {called_name: [call_locations]})."""
    defs: dict[str, str] = {}
    calls: dict[str, list[str]] = collections.defaultdict(list)
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(PACKAGE.parent)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_guard(node.name):
                defs[node.name] = f"{rel}:{node.lineno}"
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name:
                    calls[name].append(f"{rel}:{node.lineno}")
    return defs, calls


def _live_call_sites(sites: list[str]) -> list[str]:
    """Call sites that can actually execute: not tests, not dead modules."""
    return [
        c for c in sites
        if not c.startswith("tests/") and not any(c.startswith(d) for d in DEAD_MODULES)
    ]


def _unreachable():
    defs, calls = _definitions_and_call_sites()
    return {
        name: loc
        for name, loc in defs.items()
        if name not in EXEMPT and not _live_call_sites(calls.get(name, []))
    }


# --------------------------------------------------------------- the detector

def test_detector_finds_a_known_wired_guard_reachable():
    """Known-positive control: a guard that IS wired must not be flagged."""
    defs, calls = _definitions_and_call_sites()
    assert "assert_cross_arm_identity" in defs, "control guard disappeared; update this test"
    assert calls.get("assert_cross_arm_identity"), (
        "the detector found no call site for a guard that is demonstrably called "
        "from aggregate.main(); the detector is broken, not the code"
    )
    assert "assert_cross_arm_identity" not in _unreachable()


def test_detector_flags_a_known_unwired_guard(tmp_path, monkeypatch):
    """Known-negative control: a guard that is NOT wired must be flagged."""
    pkg = tmp_path / "csbench"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "def assert_never_called(x):\n    raise ValueError(x)\n\n"
        "def assert_actually_called(x):\n    raise ValueError(x)\n\n"
        "def run():\n    assert_actually_called(1)\n"
    )
    monkeypatch.setattr(pathlib.Path, "cwd", lambda: tmp_path)
    global PACKAGE
    original = PACKAGE
    try:
        PACKAGE = pkg
        unreachable = _unreachable()
        assert "assert_never_called" in unreachable, "detector missed an unwired guard"
        assert "assert_actually_called" not in unreachable, "detector false-positived"
    finally:
        PACKAGE = original


# --------------------------------------------------------------- the rule

def test_every_guard_is_reachable_from_a_production_path():
    unreachable = _unreachable()
    assert not unreachable, (
        "guards defined but never called outside tests:\n  "
        + "\n  ".join(f"{n}  ({loc})" for n, loc in sorted(unreachable.items()))
        + "\n\nA guard that only runs in its own unit tests protects nothing. Wire it "
        "into the production path it is meant to protect, or add it to EXEMPT with "
        "a reason."
    )


def test_exemptions_carry_a_reason():
    for name, reason in EXEMPT.items():
        assert reason and len(reason) > 20, f"EXEMPT[{name!r}] needs a real justification"


@pytest.mark.parametrize("guard", ["assert_item_identity", "forbid_index_keying"])
def test_item_identity_guards_are_wired_into_the_harness(guard):
    """These two are why this test exists; assert their call sites explicitly."""
    _, calls = _definitions_and_call_sites()
    sites = _live_call_sites(calls.get(guard, []))
    assert sites, f"{guard} is not called from any production path"
    assert any("harness.py" in c for c in sites), (
        f"{guard} must be wired into the RULER harness, the path where the "
        f"2026-07-17 defect occurred; found call sites: {sites}"
    )
