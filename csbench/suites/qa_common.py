"""Shared, engine-free surface for the short-answer QA suites.

This module carries the pieces that SQuAD v2, BFCL, and TruthfulQA all reuse:
the ``EvalCase`` / ``CaseResult`` records, the exact-match / F1 scorers and their
answer-normalization, the answer prompt template, a generic HTTP JSON fetch
helper, a deterministic token estimate, and the per-suite output writers and
markdown summary.

The shape mirrors a before/after context-reduction comparison: each case runs a
query against original context and against reduced context, and preservation is
scored by comparing the two answers (or, for tool-calling, against public
ground-truth arguments). Nothing here depends on any selection engine, model
client, or benchmark driver -- only the standard library. Arms are passed in by
the per-suite modules and the driver; this module never selects context itself.
"""

from __future__ import annotations

import json
import statistics
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    dataset: str
    context: str
    query: str
    ground_truth: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    dataset: str
    mode: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    fallback_used: bool
    selection_safe: bool | None
    fallback_reason: str
    selected_records: int
    available_records: int
    exact_match: bool
    f1_score: float
    contains_ground_truth: bool | None
    judge_score: float | None
    judge_reasoning: str
    accuracy_preserved: bool
    latency_original_ms: float
    latency_compressed_ms: float
    response_original: str
    response_compressed: str


def build_prompt(context: str, query: str) -> str:
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""


def estimate_context_tokens(text: str) -> int:
    """Deterministic token estimate used for the before/after size and
    reduction-ratio columns.

    This is the character-heuristic (``len // 4``) used to compute the
    published QA reduction numbers, kept exactly so those figures reproduce.
    It is intentionally separate from ``csbench.tokenizing.estimate_tokens``,
    whose lexical estimate differs.
    """
    return max(0, len(text or "") // 4)


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def compute_exact_match(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)


def compute_f1(a: str, b: str) -> float:
    a_tokens = normalize_answer(a).split()
    b_tokens = normalize_answer(b).split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in a_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in b_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def _download_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
        return rows


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    passed = sum(1 for result in results if result.accuracy_preserved)
    original_tokens = sum(result.original_tokens for result in results)
    compressed_tokens = sum(result.compressed_tokens for result in results)
    ratios = [result.compression_ratio for result in results]
    return {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "accuracy_preservation_rate": passed / len(results) if results else 0.0,
        "avg_compression_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "total_original_tokens": original_tokens,
        "total_compressed_tokens": compressed_tokens,
        "total_tokens_saved": original_tokens - compressed_tokens,
        "fallbacks": sum(1 for result in results if result.fallback_used),
        "avg_f1": statistics.fmean(result.f1_score for result in results) if results else 0.0,
        "avg_latency_compressed_ms": statistics.fmean(result.latency_compressed_ms for result in results) if results else 0.0,
    }


def write_suite_outputs(output_dir: Path, suite_name: str, results: list[CaseResult]) -> dict[str, Any]:
    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    for index, row in enumerate(rows, start=1):
        (suite_dir / f"{index:03d}_{row['case_id']}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (suite_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    summary = summarize(results)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def summary_markdown(run_summary: dict[str, Any]) -> str:
    lines = [
        "# Needlepath Before/After Context-Reduction Benchmark",
        "",
        f"Model: `{run_summary['model']}`",
        f"Judge model: `{run_summary['judge_model']}`",
        f"Limit per suite: `{run_summary['limit']}`",
        "",
        "| Benchmark | N | Accuracy Preserved | Avg Compression | Fallbacks | Tokens Saved | Read |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for suite, summary in run_summary["suites"].items():
        read = "PASS" if summary["accuracy_preservation_rate"] >= 0.90 else "CHECK"
        lines.append(
            "| {suite} | {cases} | {acc:.1%} | {comp:.1%} | {fallbacks} | {saved} | {read} |".format(
                suite=suite,
                cases=summary["cases"],
                acc=summary["accuracy_preservation_rate"],
                comp=summary["avg_compression_ratio"],
                fallbacks=summary["fallbacks"],
                saved=summary["total_tokens_saved"],
                read=read,
            )
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- SQuAD v2 uses the before/after pattern: original-context answer vs Needlepath-reduced-context answer.",
            "- BFCL uses ground-truth mode: the reduced-context answer scored against the public BFCL expected function-call values.",
            "- Needlepath fail-open is enabled: unsafe or low-confidence selections are sent as original context and counted as fallbacks.",
        ]
    )
    return "\n".join(lines) + "\n"
