"""Deterministic, dependency-free token estimate shared by all arms.

Every arm counts tokens with the same function so before/after sizes and cost
tables are comparable across methods. This is a heuristic, not a model
tokenizer - that is deliberate: it keeps the harness reproducible with no model
dependency and no per-provider drift.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: object) -> int:
    """Deterministic token estimate used for budgeting without model deps."""
    if text is None:
        return 0
    raw = str(text)
    if not raw:
        return 0
    # Word/punctuation estimate plus a char fallback for long compact strings.
    lexical = len(_WORD_RE.findall(raw))
    char_based = max(1, len(raw) // 4)
    return max(lexical, char_based)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # Approximate 4 chars/token, then refine downward.
    limit = max(64, max_tokens * 4)
    truncated = text[:limit].rstrip()
    while estimate_tokens(truncated) > max_tokens and len(truncated) > 64:
        truncated = truncated[: int(len(truncated) * 0.85)].rstrip()
    return truncated + "\n...[truncated]"
