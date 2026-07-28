"""Regression tests for the RULER item-identity contract.

These are the tests that would have caught the 2026-07-17 replication defect at
write time. `test_published_replication_artifact_fails_identity` is expected to
FAIL against the damaged artifacts and is skipped when they are absent; it is
kept as executable documentation of what the defect looks like.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csbench.suites.ruler.pairing import (
    ItemIdentityError,
    assert_item_identity,
    dedup_benign_copies,
    forbid_index_keying,
    pair_rows,
)


def row(position: int, index: int, **kw):
    return {"position": position, "index": index, "score": 1.0, "correct": True, **kw}


# --------------------------------------------------------------- identity

def test_clean_row_set_passes():
    assert_item_identity([row(i, 1000 + i) for i in range(1, 101)], source="t", expected_n=100)


def test_duplicate_position_is_rejected():
    rows = [row(1, 10), row(2, 20), row(2, 20)]
    with pytest.raises(ItemIdentityError, match="duplicate position 2"):
        assert_item_identity(rows, source="t")


def test_overwrite_shape_is_rejected():
    """The exact shape of the real defect: 100 rows, position 7 absent, 27 twice.

    Both faults are present; the duplicate is reported first because it is the
    more specific diagnosis (an evaluation was overwritten, not merely dropped).
    """
    rows = [row(i, 1000 + i) for i in range(1, 101) if i != 7]
    rows.append(row(27, 1027))
    assert len(rows) == 100
    with pytest.raises(ItemIdentityError, match="duplicate position 27"):
        assert_item_identity(rows, source="cell", expected_n=100)


def test_missing_position_alone_is_rejected():
    """Row count short of n, no duplicates: reported as absent positions."""
    rows = [row(i, 1000 + i) for i in range(1, 101) if i != 7]
    with pytest.raises(ItemIdentityError, match="positions absent"):
        assert_item_identity(rows, source="cell", expected_n=100)


def test_short_run_is_rejected_against_expected_n():
    with pytest.raises(ItemIdentityError, match="positions absent"):
        assert_item_identity([row(i, i) for i in range(1, 51)], source="t", expected_n=100)


def test_missing_required_fields():
    with pytest.raises(ItemIdentityError, match="'position'"):
        assert_item_identity([{"index": 1}], source="t")
    with pytest.raises(ItemIdentityError, match="'index'"):
        assert_item_identity([{"position": 1}], source="t")


# --------------------------------------------------------------- tripwire

def test_tripwire_allows_positions():
    forbid_index_keying(range(1, 101), source="t", n=100)


def test_tripwire_allows_a_sparse_targeted_subset():
    """Regression: an earlier version inferred the bound and rejected this."""
    forbid_index_keying(range(90, 101), source="t", n=100)


def test_tripwire_allows_a_single_position():
    forbid_index_keying([1], source="t", n=100)


def test_tripwire_rejects_ruler_index_values():
    with pytest.raises(ItemIdentityError, match="outside the valid position range"):
        forbid_index_keying([2169, 3167, 26830], source="resume cache", n=100)


def test_tripwire_rejects_a_small_index_set_that_the_old_heuristic_accepted():
    """Indices 1-3 with n=100 are in range, but 300 is not: the bound is explicit."""
    with pytest.raises(ItemIdentityError, match="outside the valid position range"):
        forbid_index_keying([1, 2, 300], source="t", n=100)


def test_tripwire_is_noop_on_empty():
    forbid_index_keying([], source="t", n=100)


# --------------------------------------------------------------- pairing

def test_pairing_aligns_by_position():
    a = [row(2, 20), row(1, 10)]
    b = [row(1, 10), row(2, 20)]
    positions, aa, bb = pair_rows(a, b)
    assert positions == [1, 2]
    assert [r["index"] for r in aa] == [10, 20]
    assert [r["index"] for r in bb] == [10, 20]


def test_pairing_refuses_to_intersect_away_a_missing_position():
    """A shrinking n_paired must be an error, never a silent smaller population."""
    a = [row(1, 10), row(2, 20)]
    b = [row(1, 10)]
    with pytest.raises(ItemIdentityError, match="position sets differ"):
        pair_rows(a, b)


def test_pairing_detects_same_position_different_item():
    a = [row(1, 10)]
    b = [row(1, 99)]
    with pytest.raises(ItemIdentityError, match="index identity mismatch"):
        pair_rows(a, b)


def test_pairing_rejects_duplicate_position_inputs():
    with pytest.raises(ItemIdentityError, match="duplicate position"):
        pair_rows([row(1, 10), row(1, 10)], [row(1, 10)])


# --------------------------------------------------------------- dedup

def test_dedup_collapses_identical_copies():
    rows = [row(1, 10), row(2, 20), row(2, 20), row(2, 20)]
    out, removed = dedup_benign_copies(rows, source="t")
    assert removed == 2
    assert [r["position"] for r in out] == [1, 2]


def test_dedup_refuses_differing_copies():
    """Two genuinely different evaluations at one position is unrecoverable."""
    rows = [row(1, 10, score=1.0), row(1, 10, score=0.0)]
    with pytest.raises(ItemIdentityError, match="DIFFERING content"):
        dedup_benign_copies(rows, source="t")


def test_dedup_does_not_invent_missing_positions():
    """Dedup recovers survivors; it never restores what the overwrite destroyed."""
    rows = [row(i, 1000 + i) for i in range(1, 100) if i != 7] + [row(27, 1027)]
    out, removed = dedup_benign_copies(rows, source="t")
    assert removed == 1
    assert 7 not in {r["position"] for r in out}
    with pytest.raises(ItemIdentityError):
        assert_item_identity(out, source="t", expected_n=99)


# --------------------------------------------------------------- artifacts

_REPL = Path(__file__).resolve().parents[1] / "runs" / "ruler_repl_v1" / "items"


@pytest.mark.skipif(not _REPL.exists(), reason="published replication artifacts not present")
def test_published_replication_artifact_fails_identity():
    """Documents the defect: the published cells violate the contract."""
    failures = []
    for path in sorted(_REPL.glob("*/*/*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        try:
            assert_item_identity(rows, source=str(path), expected_n=100)
        except ItemIdentityError:
            failures.append(path)
    assert len(failures) == 42, f"expected the 42 known-damaged cells, got {len(failures)}"


# --------------------------------------------------------------- input hash

def test_input_sha256_is_stable_and_content_sensitive():
    from csbench.suites.ruler.data import OfficialRulerExample
    from csbench.suites.ruler.harness import input_sha256

    def ex(text):
        return OfficialRulerExample(index=1, input_text=text, answer_prefix="",
                                    expected_answer=("a",), length=10, task="t")

    a, b = ex("haystack with a needle"), ex("haystack with a needle")
    assert input_sha256(a) == input_sha256(b), "same input must hash the same"
    assert input_sha256(a) != input_sha256(ex("haystack with a neeble")), (
        "a one-character change must change the hash; this is the whole point of "
        "recording it"
    )
    assert len(input_sha256(a)) == 64


def test_rows_carry_an_input_hash():
    """The artifact-schema gap: the 2026-07-17 deposit recorded no input hash,
    so byte-identity could not be checked retrospectively."""
    import inspect
    from csbench.suites.ruler import harness
    src = inspect.getsource(harness)
    assert '"input_sha256": input_sha256(example),' in src, (
        "per-row input hash is not stamped onto the row; a future regeneration "
        "would again be unverifiable byte-for-byte"
    )
