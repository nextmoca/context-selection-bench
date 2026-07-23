"""RULER dataset loading and RULER-style string-match scoring.

This module provides the generic, engine-independent surface for the RULER
long-context suite: loading official RULER JSONL examples, splitting an example
into context and question, the RULER string-matching scorer, and small text
utilities (keyword extraction, chunking) used to build selection inputs.

It has no dependency on any selection engine, model client, or benchmark driver
-- only the standard library plus ``csbench.tokenizing``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


STOPWORDS = {
    "all",
    "and",
    "are",
    "assigned",
    "for",
    "find",
    "following",
    "in",
    "is",
    "mentioned",
    "number",
    "numbers",
    "provided",
    "question",
    "special",
    "text",
    "the",
    "to",
    "value",
    "what",
}


@dataclass(frozen=True)
class OfficialRulerExample:
    task: str
    index: int
    input_text: str
    answer_prefix: str
    expected_answer: Sequence[str]
    length: int


def match_type_for_task(task: str) -> str:
    return "part" if task.startswith("qa_") else "all"


def score_ruler_answer(answer: str, expected: Iterable[str], match_type: str) -> float:
    prediction = re.sub(r"[\x00-\x1f]", "\n", answer or "").strip().lower()
    references = [str(item).lower() for item in expected]
    if not references:
        return 0.0
    if match_type == "part":
        return max((1.0 if reference in prediction else 0.0) for reference in references)
    return sum(1.0 if reference in prediction else 0.0 for reference in references) / len(references)


def split_official_prompt(input_text: str) -> tuple[str, str]:
    stripped = (input_text or "").strip()
    question_markers = ["\nQuestion:", "\nWhat "]
    best_idx = -1
    for marker in question_markers:
        idx = stripped.rfind(marker)
        if idx > best_idx:
            best_idx = idx
    if best_idx >= 0:
        return stripped[:best_idx].strip(), stripped[best_idx:].strip()
    lines = stripped.splitlines()
    if len(lines) <= 1:
        return stripped, stripped
    return "\n".join(lines[:-1]).strip(), lines[-1].strip()


def selection_keywords_for_example(example: OfficialRulerExample, question: str) -> tuple[str, ...]:
    gold = {str(item).lower() for item in example.expected_answer}
    text = f"{question} {example.answer_prefix}"
    terms: list[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        token = raw_token.lower()
        if len(token) < 3 or token in STOPWORDS or token in gold:
            continue
        if token.isdigit() and token in gold:
            continue
        terms.append(token)
        if "-" in token:
            terms.extend(part for part in token.split("-") if len(part) >= 3 and part not in STOPWORDS)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        deduped.append(term)
        seen.add(term)
    return tuple(deduped[:24])


def chunk_text(text: str, *, target_tokens: int = 260, overlap_tokens: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    target_words = max(80, target_tokens * 3 // 4)
    overlap_words = max(0, overlap_tokens * 3 // 4)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def load_examples(data_root: Path, task: str, *, limit: int) -> list[OfficialRulerExample]:
    path = data_root / task / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"RULER task data not found: {path}")
    examples: list[OfficialRulerExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                OfficialRulerExample(
                    task=task,
                    index=int(raw["index"]),
                    input_text=str(raw["input"]),
                    answer_prefix=str(raw.get("answer_prefix") or ""),
                    expected_answer=tuple(str(item) for item in raw["outputs"]),
                    length=int(raw.get("length") or 0),
                )
            )
            if limit and len(examples) >= limit:
                break
    return examples
