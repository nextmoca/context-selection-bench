from csbench.tokenizing import estimate_tokens, truncate_to_token_budget


def test_estimate_tokens_golden():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("hello world") == 2
    # punctuation counts as separate tokens
    assert estimate_tokens("a, b.") == 4
    # char fallback dominates for a long unspaced run
    assert estimate_tokens("x" * 40) == 10


def test_truncate_to_token_budget():
    assert truncate_to_token_budget("anything", 0) == ""
    short = "hello world"
    assert truncate_to_token_budget(short, 100) == short
    long = " ".join(["word"] * 500)
    out = truncate_to_token_budget(long, 20)
    assert estimate_tokens(out) <= 20 + estimate_tokens("\n...[truncated]")
    assert out.endswith("[truncated]")
