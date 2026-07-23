"""Deterministic scorer coverage for the SQuAD v2 / BFCL / TruthfulQA suites.

These suites' per-item run artifacts are not committed, so the byte-parity
surface here is the ported scorer/parse functions themselves, exercised on
hand-built inputs with known outputs.
"""

import math

from csbench.suites.bfcl import check_bfcl_ground_truth
from csbench.suites.qa_common import compute_exact_match, compute_f1, normalize_answer
from csbench.suites.truthfulqa import parse_mc1_choice


def test_normalize_answer_strips_articles_punctuation_case():
    assert normalize_answer("The quick, brown fox!") == "quick brown fox"
    assert normalize_answer("An Apple") == "apple"
    assert normalize_answer("") == ""


def test_exact_match_after_normalization():
    assert compute_exact_match("The Cat.", "a cat") is True
    assert compute_exact_match("cat", "dog") is False


def test_f1_token_overlap():
    assert compute_f1("the cat", "a cat") == 1.0  # both normalize to "cat"
    assert math.isclose(compute_f1("cat sat", "cat"), 2 / 3)  # p=1/2, r=1/1
    assert compute_f1("", "") == 1.0
    assert compute_f1("cat", "") == 0.0
    assert compute_f1("dog", "cat") == 0.0


def test_bfcl_ground_truth_matching():
    gt = '[{"get_weather": {"city": ["Paris"], "unit": ["celsius"]}}]'
    assert check_bfcl_ground_truth("weather in Paris, celsius please", gt) is True
    assert check_bfcl_ground_truth("weather in London", gt) is False  # city value absent
    assert check_bfcl_ground_truth("anything", "not json") is False


def test_bfcl_numeric_variant_matching():
    gt = '[{"set_temp": {"value": [72]}}]'
    assert check_bfcl_ground_truth("set it to 72 degrees", gt) is True
    assert check_bfcl_ground_truth("set it to 71 degrees", gt) is False


def test_truthfulqa_mc1_parse():
    assert parse_mc1_choice("A", 4) == "A"
    assert parse_mc1_choice("(B)", 4) == "B"
    assert parse_mc1_choice("The answer is C.", 4) == "C"
    assert parse_mc1_choice("D", 3) is None  # out of range for 3 choices
    assert parse_mc1_choice("", 4) is None
    assert parse_mc1_choice("could be A or B honestly", 4) is None  # ambiguous


def test_suite_modules_import_without_datasets_at_module_load():
    # squad/truthfulqa lazy-import `datasets` inside their loaders, so importing
    # the module must not require the optional 'qa' extra.
    import importlib

    for mod in ("csbench.suites.squad", "csbench.suites.truthfulqa", "csbench.suites.bfcl"):
        importlib.import_module(mod)
