"""Assert the paper matches the frozen headline figures.

Scope limit, stated up front because it is easy to over-read a green run:
this proves the paper agrees with `tests/frozen_figures.py`, which is a DERIVED
COPY of the tracker registry. It proves nothing about the website, which keeps
its own copy behind its own assertions and cannot be imported from here.
Cross-repository agreement is a human reconciliation step. See the header of
`frozen_figures.py`.

The detector is validated against a known-positive and a known-negative before
it is trusted, per standing program rule.
"""
from __future__ import annotations

import math
import pathlib

import pytest

from tests.frozen_figures import (
    FROZEN_FIGURES,
    SUPERSEDED,
    SUPERSEDED_ALLOWANCE,
    SUPERSEDED_INTERVALS,
    UNROUNDED_CI,
)

PAPER = pathlib.Path(__file__).resolve().parents[1] / "paper" / "main.tex"


def paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


# --------------------------------------------------------------- detector

def test_detector_finds_a_string_known_to_be_present():
    """Known-positive control: a literal that is definitely in the paper."""
    assert r"$+4.62\pp$" in paper_text(), (
        "the corrected pooled delta is absent from the paper; either the paper "
        "regressed or this control is stale"
    )


def test_detector_does_not_find_a_string_known_to_be_absent():
    """Known-negative control: the detector must be able to report absence."""
    assert r"$+9.99\pp$" not in paper_text(), "control string unexpectedly present"


# --------------------------------------------------------------- the rule

@pytest.mark.parametrize("ident,literal,description", FROZEN_FIGURES,
                         ids=[f[0] for f in FROZEN_FIGURES])
def test_frozen_figure_appears_in_the_paper(ident, literal, description):
    assert literal in paper_text(), (
        f"frozen figure {ident} ({description}) is missing from paper/main.tex.\n"
        f"Expected literal: {literal}\n\n"
        "This figure is published on a live claim surface. The paper must match "
        "it; it does not get adjusted to match the paper. If the figure genuinely "
        "needs to change, that is a coordinated correction across the site, the "
        "PDFs and the Zenodo record, and it is sequenced by the maintainer, not "
        "resolved by editing this test."
    )


@pytest.mark.parametrize("literal,description", SUPERSEDED,
                         ids=[s[1][:40] for s in SUPERSEDED])
def test_superseded_figures_do_not_reappear(literal, description):
    count = paper_text().count(literal)
    allowed = SUPERSEDED_ALLOWANCE.get(literal, 0)
    assert count <= allowed, (
        f"superseded value {literal} ({description}) appears {count} time(s), "
        f"allowance {allowed}. A pre-correction figure has survived a sweep or "
        "an edit reverted."
    )


def test_manifest_declares_itself_derived():
    """The file must not read as self-certifying to a future editor."""
    header = (pathlib.Path(__file__).resolve().parent / "frozen_figures.py").read_text()
    for phrase in ("DERIVED COPY", "tracker", "Neither consumer can import the other",
                   "Reconciliation between the two copies is a HUMAN step",
                   "Last reconciled"):
        assert phrase in header, (
            f"frozen_figures.py header lost its statement of limits ({phrase!r}). "
            "A reader must not be able to conclude these numbers are "
            "self-certifying."
        )


# --------------------------------------------------------------- rounding

def _outward(low: float, high: float, places: int = 1) -> tuple[float, float]:
    """Round an interval OUTWARD: floor the lower, ceil the upper.

    The published interval must contain the computed one. Rounding to nearest
    can move a bound inward and publish an interval narrower than the data
    supports.
    """
    scale = 10 ** places
    return math.floor(low * scale) / scale, math.ceil(high * scale) / scale


@pytest.mark.parametrize("ident", sorted(UNROUNDED_CI))
def test_published_bounds_are_the_outward_rounding_of_the_computed_ones(ident):
    low, high, pub_low, pub_high = UNROUNDED_CI[ident]
    exp_low, exp_high = _outward(low, high)
    assert (pub_low, pub_high) == pytest.approx((exp_low, exp_high)), (
        f"{ident}: published [{pub_low}, {pub_high}] is not the outward rounding "
        f"of computed [{low}, {high}], which is [{exp_low}, {exp_high}].\n"
        "Interval bounds round OUTWARD (floor lower, ceil upper) so the published "
        "interval contains the computed one. If this looks off by one in the last "
        "digit, that is the convention working, not a typo."
    )


@pytest.mark.parametrize("ident", sorted(UNROUNDED_CI))
def test_published_interval_contains_the_computed_interval(ident):
    """The property the convention exists to guarantee, asserted directly."""
    low, high, pub_low, pub_high = UNROUNDED_CI[ident]
    assert pub_low <= low, f"{ident}: published lower {pub_low} is INWARD of computed {low}"
    assert pub_high >= high, f"{ident}: published upper {pub_high} is INWARD of computed {high}"


def test_rounding_helper_rounds_outward_not_to_nearest():
    """Known-positive/negative control for the helper itself."""
    # to-nearest would give (2.6, 6.4); outward must give (2.5, 6.4)
    assert _outward(2.5900, 6.3595) == pytest.approx((2.5, 6.4))
    # to-nearest would give (3.3, 5.9); outward must give (3.3, 6.0)
    assert _outward(3.3178, 5.9394) == pytest.approx((3.3, 6.0))
    # an exact bound must not be widened spuriously
    assert _outward(2.0, 6.0) == pytest.approx((2.0, 6.0))


@pytest.mark.parametrize("literal,description", SUPERSEDED_INTERVALS,
                         ids=[s[1][:34] for s in SUPERSEDED_INTERVALS])
def test_inward_rounded_intervals_do_not_reappear(literal, description):
    assert literal not in paper_text(), (
        f"interval {literal} ({description}) is narrower than the data supports "
        "and was corrected on 2026-07-28; it must not return."
    )
