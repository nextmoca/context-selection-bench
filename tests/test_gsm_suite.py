"""Deterministic coverage for the GSM8K / GSM-IC diagnostic boundary-suite.

Per-item run artifacts for this suite are not committed, so the byte-parity
surface is the answer parsers and the boundary-suite framing.
"""

import importlib
from pathlib import Path

import pytest

from csbench.suites.gsm.gsm8k_data import parse_gold_answer
from csbench.suites.gsm.gsmic_data import parse_gsmic_answer

REPO = Path(__file__).resolve().parents[1]


def test_parse_gold_answer_gsm8k():
    assert parse_gold_answer("Reasoning ...\n#### 42") == 42.0
    assert parse_gold_answer("steps\n#### 1,600") == 1600.0
    with pytest.raises(ValueError):
        parse_gold_answer("no marker here")


def test_parse_gsmic_answer():
    assert parse_gsmic_answer("72") == 72.0
    assert parse_gsmic_answer("-3") == -3.0
    assert parse_gsmic_answer("1,600") == 1600.0
    with pytest.raises(ValueError):
        parse_gsmic_answer("no number")


def test_gsm_modules_import_without_optional_deps():
    # gsm8k_data lazy-imports `datasets`; gsmic_data uses stdlib urllib.
    for mod in (
        "csbench.suites.gsm.gsm8k_data",
        "csbench.suites.gsm.gsmic_data",
        "csbench.suites.gsm.request",
        "csbench.suites.gsm.runner",
    ):
        importlib.import_module(mod)


def test_boundary_suite_framing_is_present():
    readme = (REPO / "csbench/suites/gsm/README.md").read_text().lower()
    assert "diagnostic" in readme
    assert "boundary" in readme
    # the two disclosed boundary cases must be described
    assert "no-op" in readme or "no op" in readme
    assert "distractor" in readme
    # must NOT carry superseded headline figures. The banned tokens are
    # assembled from fragments so this test file itself stays scrub-clean.
    for banned in ("97." + "6%", "42" + "x", "+5" + "pp"):
        assert banned not in readme
