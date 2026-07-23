import math

from csbench.contracts import ContextResponse
from csbench.stats import (
    bootstrap_ci_delta,
    fallback_rate,
    macro_accuracy,
    mcnemar_test,
    parse_final_numeric_answer,
    score_exact_match,
)


def test_parse_and_exact_match():
    assert parse_final_numeric_answer("blah #### 42") == 42.0
    assert parse_final_numeric_answer("the total is 17 apples") == 17.0  # last-number fallback
    assert parse_final_numeric_answer("no numbers here") is None
    assert score_exact_match("reasoning ... #### 42", 42.0) is True
    assert score_exact_match("reasoning ... #### 41", 42.0) is False
    assert score_exact_match("i refuse", 42.0) is False
    assert score_exact_match("#### 1,234", 1234.0) is True  # comma stripping


def test_mcnemar_identical_arrays_have_no_discordance():
    a = [True, False, True, True]
    r = mcnemar_test(a, a)
    assert r.b == 0 and r.c == 0
    assert r.p_value == 1.0
    assert r.n_items == 4


def test_mcnemar_all_discordant():
    r = mcnemar_test([True] * 10, [False] * 10)
    assert sorted([r.b, r.c]) == [0, 10]
    assert r.p_value < 0.01


def test_bootstrap_ci_delta_is_deterministic_and_separated():
    a, b = [True] * 20, [False] * 20
    r = bootstrap_ci_delta(a, b, n_resamples=1000, seed=42)
    assert math.isclose(r.observed_delta, 1.0)
    assert r.n_items == 20
    # every paired difference is exactly 1.0, so the CI collapses to a point
    assert r.ci_low == 1.0 and r.ci_high == 1.0
    r2 = bootstrap_ci_delta(a, b, n_resamples=1000, seed=42)
    assert (r2.ci_low, r2.ci_high, r2.observed_delta) == (r.ci_low, r.ci_high, r.observed_delta)


def test_macro_accuracy_base_problem_all_correct_semantics():
    r = macro_accuracy([(0, True), (0, False), (1, True)])
    assert r.n_base_problems == 2
    assert r.n_items == 3
    assert math.isclose(r.accuracy, 0.5)
    assert r.failed_base_problem_ids == [0]


def test_fallback_rate_over_responses():
    resps = [
        ContextResponse(request_id="1", rendered_context="", fallback_used=True),
        ContextResponse(request_id="2", rendered_context="", fallback_used=False),
        ContextResponse(request_id="3", rendered_context="", fallback_used=False),
    ]
    assert math.isclose(fallback_rate(resps), 1 / 3)
