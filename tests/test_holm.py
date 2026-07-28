"""Tests for the Holm-Bonferroni correction the paper claims as its primary
decision rule."""
from __future__ import annotations

import pytest

from csbench.stats import holm_adjust


def by_label(results):
    return {r.label: r for r in results}


def test_family_of_two_both_reject():
    """The published replication figures: both per-length tests survive Holm."""
    r = by_label(holm_adjust({"8k": 0.0144, "16k": 0.0031}))
    assert r["16k"].threshold == pytest.approx(0.025)   # smallest vs alpha/2
    assert r["8k"].threshold == pytest.approx(0.05)     # next vs alpha/1
    assert r["16k"].reject and r["8k"].reject


def test_family_of_two_on_corrected_population():
    """The 2468-survivor figures: still both significant, but by a thin margin."""
    r = by_label(holm_adjust({"8k": 0.0305, "16k": 0.0221}))
    assert r["16k"].reject and r["8k"].reject
    assert r["16k"].p_raw < r["16k"].threshold
    assert r["16k"].threshold - r["16k"].p_raw == pytest.approx(0.0029, abs=1e-4)


def test_step_down_stops_at_first_failure():
    """Holm retains every remaining hypothesis once one fails, even a tiny p."""
    r = by_label(holm_adjust({"a": 0.30, "b": 0.04}))
    assert not r["b"].reject          # 0.04 > 0.05/2
    assert not r["a"].reject


def test_non_significant_family():
    r = by_label(holm_adjust({"8k": 0.1130, "16k": 0.7415}))
    assert not r["8k"].reject and not r["16k"].reject


def test_adjusted_p_is_monotone_and_capped():
    res = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.9})
    adj = [r.p_adjusted for r in res]
    assert adj == sorted(adj), "adjusted p-values must be non-decreasing"
    assert all(p <= 1.0 for p in adj)


def test_adjusted_p_matches_step_down_formula():
    res = by_label(holm_adjust({"a": 0.01, "b": 0.04}))
    assert res["a"].p_adjusted == pytest.approx(0.02)   # 2 * 0.01
    assert res["b"].p_adjusted == pytest.approx(0.04)   # max(0.02, 1 * 0.04)


def test_single_test_family_is_uncorrected():
    r = by_label(holm_adjust({"only": 0.04}))
    assert r["only"].threshold == pytest.approx(0.05)
    assert r["only"].p_adjusted == pytest.approx(0.04)
    assert r["only"].reject


def test_rejects_empty_family():
    with pytest.raises(ValueError, match="at least one"):
        holm_adjust({})


def test_rejects_out_of_range_p():
    with pytest.raises(ValueError, match="outside"):
        holm_adjust({"a": 1.5})
