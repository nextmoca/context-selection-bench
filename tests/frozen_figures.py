"""Frozen headline figures, as published.

WHAT THIS FILE IS
-----------------
A DERIVED COPY. It is not the source of truth and it cannot certify itself.

The source of truth is the frozen-figures registry in the program tracker.
Two consumers are registered against it:

  1. the public website  (nextmoca.com, src/data/benchmarks.ts, guarded by its
     own build-time assertions)
  2. this repository     (this file, guarded by tests/test_frozen_figures.py)

Neither consumer can import the other. One repository is public and one is
private, and no test in either can see across the boundary. So each will pass
its own build while silently disagreeing with the other, and a green suite here
is evidence about THIS repository only.

Reconciliation between the two copies is a HUMAN step, owned by the maintainer
against the tracker registry. No test performs it and no test can.

If you are editing a number in this file: you are editing a copy. Change the
tracker registry first, reconcile every consumer listed there, then update this
file and the date below. A figure that differs from the registry is wrong here
even if every test in this repository passes.

ROUNDING CONVENTION
-------------------
Point estimates round to NEAREST.

Interval bounds round OUTWARD: floor the lower bound, ceil the upper bound, so
the published interval always CONTAINS the computed one. Rounding a bound to
nearest can move it inward and publish an interval narrower than the data
supports, which overstates precision.

This is not advisory. `UNROUNDED_CI` below carries the computed bounds and
`test_frozen_figures.py` recomputes the outward rounding from them and asserts
the published literals match. If you change a bound, the recomputation must
agree or the build fails. Do not "correct" an outward-rounded bound back to
nearest because it looks off by one in the last digit; that is the convention
working.

Last reconciled against the tracker registry: 2026-07-28
Figures below are the corrected v1.1 values on the 2,468-item population.
"""
from __future__ import annotations

# Each entry: (identifier, LaTeX-escaped literal that must appear in the paper,
#              human description). The literal is matched verbatim against
#              paper/main.tex, so a rounding change breaks the build.
FROZEN_FIGURES: list[tuple[str, str, str]] = [
    ("np_pooled_delta", r"$+4.62\pp$",
     "Needlepath vs full context, pooled accuracy delta"),
    ("np_pooled_ci", r"$[+3.32,+5.94]$",
     "pooled accuracy delta 95% CI, unrounded (site renders outward-rounded "
     "one decimal: [+3.3, +6.0])"),
    ("np_token_reduction", r"$32.7\%$",
     "mean input-token reduction, 11,788 -> 7,929 (exact 32.737%)"),
    ("np_cost", r"\$15.67",
     "Needlepath end-to-end cost on the corrected population"),
    ("full_cost", r"\$23.29",
     "full-context cost on the corrected population"),
    ("compresr_cost", r"\$20.70",
     "Compresr cost: $18.03 API + $2.68 compression fee "
     "(COMPRESR_RESULTS.md:111); used as given, not re-derived"),
    ("compresr_token_reduction", r"$-22.6\%$",
     "Compresr mean input-token reduction (exact 22.616%)"),
    ("np_8k_delta", r"$+4.48\pp$", "Needlepath vs full context at 8K"),
    ("np_8k_ci_paper", r"$[+2.59,+6.36]$",
     "8K accuracy delta CI, unrounded as carried in the paper "
     "(site renders outward-rounded [+2.5, +6.4])"),
    ("np_16k_delta", r"$+4.77\pp$", "Needlepath vs full context at 16K"),
    ("np_16k_ci_paper", r"$[+2.87,+6.67]$",
     "16K accuracy delta CI, unrounded as carried in the paper "
     "(site renders outward-rounded [+2.8, +6.7])"),
    ("np_selection_latency_low", r"$8.5$", "Needlepath in-process selection, low end (ms)"),
    ("np_selection_latency_high", r"$14.5$", "Needlepath in-process selection, high end (ms)"),
]

# Computed (unrounded) CI bounds, in percentage points, and the outward-rounded
# one-decimal form published on the claim surfaces. The test recomputes the
# right-hand column from the left and asserts agreement, so the convention is
# enforced rather than described.
#   identifier -> (computed_low, computed_high, published_low, published_high)
UNROUNDED_CI: dict[str, tuple[float, float, float, float]] = {
    "np_8k_ci":     (2.5900, 6.3595, 2.5, 6.4),
    "np_16k_ci":    (2.8685, 6.6680, 2.8, 6.7),
    "np_pooled_ci": (3.3178, 5.9394, 3.3, 6.0),
}


# Values that must NOT appear: superseded figures from the damaged population.
# These are the pre-correction numbers; their presence means an edit reverted
# or a stale passage survived a sweep.
SUPERSEDED: list[tuple[str, str]] = [
    (r"$+4.99\pp$", "pre-correction pooled delta (damaged 2,600-row population)"),
    (r"$[+3.66,+6.32]$", "pre-correction pooled CI"),
    (r"$+4.64\pp$", "pre-correction 8K delta"),
    (r"$+5.33\pp$", "pre-correction 16K delta"),
    (r"$+2.50\pp$", "pre-correction binarized delta"),
    (r"\$24.53", "pre-correction full-context cost (raw population)"),
]

# Interval bounds that were published before the outward-rounding rule and were
# narrower than the data supports. Corrected 2026-07-28 (registry + site PR #1).
SUPERSEDED_INTERVALS: list[tuple[str, str]] = [
    ("[+3.3, +5.9]", "pooled CI with upper rounded inward (computed 5.9394)"),
    ("[+2.6, +6.4]", "8K CI with lower rounded inward (computed 2.5900)"),
    ("[+2.9, +6.7]", "16K CI with lower rounded inward (computed 2.8685)"),
]

# The superseded pooled delta legitimately appears once, in the disclosure
# section that explains the correction ("from +4.99pp on the damaged population
# to +4.62pp"). Occurrences beyond this budget are stale text.
SUPERSEDED_ALLOWANCE: dict[str, int] = {
    r"$+4.99\pp$": 1,
}
