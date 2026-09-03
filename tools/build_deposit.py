#!/usr/bin/env python3
"""Assemble a license-gated, scrub-clean public deposit for a benchmark run.

The per-run policy is documented in ``DEPOSIT.md`` and summarized in
``RELEASE.md``. Safety model:

  * STRICT ALLOWLIST for per-item fields: unknown fields are dropped, never
    shipped, so a new field cannot silently leak.
  * HARD EXCLUDES, always: gate engine-internal signals (any ``signals`` path,
    and the per-item relevance-gap / gate-score / raw-reason fields), prompt /
    context / haystack / trajectory / history text, the AppWorld dataset canary,
    and (in aggregates) host/filesystem and internal-identifier strings.
  * The output directory is re-checked with ``scripts/scrub_check.py`` (which
    also catches the canary and any residual internal id); the build FAILS if
    the gate finds anything.

To avoid duplicating internal-identifier literals in this (scanned) file, the
redaction patterns are imported from the scrub gate itself.

Per-run types:
  ruler          RULER-family outputs (ruler_v1, ruler_repl_v1). Per-item
                 OUTPUTS only (completions, scores, token counts, metadata);
                 prompts are never present/shipped; qa_1/qa_2 answers (SQuAD/
                 HotpotQA-derived, CC BY-SA) are hashed. Handles both the
                 items/**/*.jsonl and per-item <task>/NNN_i.json layouts.
  gate_ruler     RULER gate regression (gate_ruler_v1): per-row engage OUTCOME
                 + baseline/needlepath scores. The gate-internal signal values
                 (gap, gate_score, raw reason) are dropped by construction.
  appworld       AppWorld family (appworld_v1, gate_appworld_v1, gate_probe_v1):
                 numeric aggregates + per-episode metrics (task_id hashed, no
                 text). Trajectories / env / llm histories are never read.
  controls_plain TruthfulQA / BFCL (Apache-2.0): upstream items + our outputs,
                 plaintext, with NOTICE + citation.
  controls_hash  SQuAD v2 (CC BY-SA): outputs + scores; query/answer/response
                 hashed (extractive spans of share-alike passages).
  gsm8k_hash     GSM8K (MIT) hash-only fallback: aggregates + manifest + a
                 regenerate-via-fetch note (no per-item run recoverable).

Usage:
  python tools/build_deposit.py --run-type ruler --run-id ruler_repl_v1 \
      --src runs/ruler_repl_v1 --out deposit
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# --- per-item field allowlist for RULER-family output records ---------------
_RULER_ITEM_ALLOW = {
    # identity / cell
    "arm", "task", "length", "length_label", "index", "position",
    "example_length", "operating_point", "policy_version",
    # scores / correctness
    "correct", "score", "coverage_score", "match_type", "evidence_shape",
    "baseline_score", "needlepath_score", "llmlingua2_score",
    # tokens / reductions
    "input_tokens", "output_tokens", "tokens_before", "tokens_after",
    "tokens_saved", "reduction_ratio", "budget_tokens", "attempted_budget_tokens",
    "selected_budget_tokens", "adaptive_budget_enabled",
    "baseline_input_tokens", "needlepath_input_tokens", "llmlingua2_input_tokens",
    "baseline_output_tokens", "needlepath_output_tokens", "llmlingua2_output_tokens",
    "state_items_available", "state_items_selected",
    "tokens_available_before_filtering", "tokens_selected_after_filtering",
    "state_token_reduction_pct", "final_input_reduction_pct",
    "llmlingua2_compression_ratio", "llmlingua2_compression_rate_reported",
    "llmlingua2_origin_tokens", "llmlingua2_compressed_tokens",
    "llmlingua2_final_input_reduction_pct",
    # selection OUTCOME metadata (not gate signals)
    "records_available", "records_selected", "selected_record_ids",
    "fallback_used", "fallback_reason", "selection_safe", "selection_error",
    "needlepath_prompt_used_full_context", "gold_in_selected_context",
    # latency
    "engine_latency_ms", "model_latency_ms", "selection_latency_ms",
    "baseline_latency_ms", "needlepath_latency_ms", "llmlingua2_latency_ms",
    "llmlingua2_compression_latency_ms",
    # verification digests: exact lowercase sha256 hex, validated in clean_ruler_record
    "item_sha256", "expected_answer_sha256", "prompt_sha256", "input_sha256",
    # format + answers (answers hashed for qa tasks; see below)
    "format_metrics", "answer", "expected_answer",
    "baseline_answer", "needlepath_answer", "llmlingua2_answer",
}
# Deliberately NOT in the allowlist (engine-internal reasoning traces, dropped):
#   missing_required_state_ids, missing_parent_state_ids, missing_obligation_ids,
#   satisfied_obligation_ids, repair_hint_reasons, qa_answerability_passed,
#   qa_support_span_count.
_QA_TASKS = {"qa_1", "qa_2"}          # embed SQuAD/HotpotQA (CC BY-SA) -> hash
_QA_HASH_FIELDS = {
    "answer", "expected_answer",
    "baseline_answer", "needlepath_answer", "llmlingua2_answer",
}

# --- RULER gate regression (outcome only; signal values dropped) ------------
_GATE_RULER_ALLOW = {
    "task", "length_label", "length", "position", "index", "engage",
    "baseline_score", "needlepath_score", "needlepath_prompt_used_full_context",
    "n_candidates",
}
# Dropped by construction: gap, gate_score, reason (gate-internal signals).

# --- control-suite per-item allowlist ---------------------------------------
_CONTROLS_ALLOW = {
    "arm", "case_id", "correct", "f1", "prompt_tokens", "completion_tokens",
    "cost_usd", "reduction_ratio", "compressed_text_len",
    "protected_record_present", "gate_status", "schema_form",
    "dataset_revision", "parsed_letter", "malformed",
}
_CONTROLS_TEXT = {"query", "ground_truth", "response"}   # plaintext or hashed
# Dropped: gate_reason (may reveal internal decision logic).

# --- AppWorld per-episode metric allowlist (numeric/categorical; no text) ----
_APPWORLD_EPISODE_ALLOW = {
    "arm", "split", "condition", "recovery_enabled", "recovery_requests",
    "recovery_hits", "recovery_misses", "success_in_run", "termination_reason",
    "iterations", "final_reward",
}
_APPWORLD_EPISODE_NUM_RE = re.compile(r"(^total_.*_tokens$)|(^total_cost_usd$)|(^needlepath_)")
# AppWorld aggregate keys that carry task-id lists / per-task detail -> drop.
_APPWORLD_DENY_KEYS = {
    "per_episode", "missing_task_ids", "ran_but_unscored_task_ids",
    "recovered_ids", "missing", "missing_ids",
}
_APPWORLD_CONFIG_ALLOW = {
    "run", "arm", "probe", "conditions", "model_name", "seed", "split", "n_tasks",
}

_CONFIG_ALLOW = {
    "run_name", "arms", "model", "temperature", "seed", "num_samples",
    "lengths", "tasks", "task_list", "leakage_controls", "scorer",
    "max_output_tokens", "operating_point",
}
# never emit a prompt-ish / trajectory field, whatever the source calls it
_PROMPTISH = {
    "prompt", "context", "haystack", "input_text", "full_context", "messages",
    "system", "user", "rendered_context", "records", "text", "input",
    "trajectory", "task_instruction", "env_history", "llm_history",
}
# path-based hard excludes: gate signals, temp/config scratch, per-task dumps
_EXCLUDE_PATH = re.compile(
    r"(^|/)(signals|_tmp[^/]*|[^/]*co_configs|task_[^/]*|full_history|acon|needlepath)(/|$)"
    r"|(trajectory|env_history|llm_history|appworld_trajectory)"
)
# The arm directory `items/<length>/needlepath/` is result data, not an engine
# scratch directory, but the `needlepath` word in the pattern above matched it and
# dropped every Needlepath per-item row from the ruler_repl_v1 deposit while the
# full_context and compresr arms shipped (ruler_v1 keeps all arms in one record
# per item and was not affected). Exactly that arm segment is masked before the
# pattern runs; every other name at arm depth (signals, _tmp*, task_*, ...) is
# still subject to the exclusion.
_ITEMS_ARM_DEPTH = 2  # items/<length>/<arm>/...
_ARM_NAMES_THAT_COLLIDE_WITH_SCRATCH = {"needlepath"}


def _excluded_path(rel: Path) -> bool:
    parts = list(rel.parts)
    if len(parts) > _ITEMS_ARM_DEPTH + 1 and parts[0] == "items" \
            and parts[_ITEMS_ARM_DEPTH] in _ARM_NAMES_THAT_COLLIDE_WITH_SCRATCH:
        parts[_ITEMS_ARM_DEPTH] = "arm"
    return bool(_EXCLUDE_PATH.search("/".join(parts)))


def _load_leak_patterns():
    """Reuse the scrub gate's sensitive-LITERAL patterns (secrets, host/filesystem
    paths, internal buckets/tickets/module-paths/worktrees/doc-paths/source-files/
    results-docs, private IPs, the AppWorld canary) plus the author-name pattern,
    so no id literals live in this file.

    The attribution/figure patterns are commit- and prose-hygiene rules for OUR
    repository, not data-leak rules: they false-positive on legitimate third-party
    benchmark text (an upstream item may coincidentally contain an ordinary English
    phrase that an attribution rule matches). They are intentionally excluded from
    the deposit's leak scan, which ships upstream data verbatim under its own
    license."""
    spec = importlib.util.spec_from_file_location(
        "_scrub_check", _REPO / "scripts" / "scrub_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    named = [(name, pat) for name, pat in mod._PATTERNS
             if not (name.startswith("attribution:") or name.startswith("figure:"))]
    named.append(("name:author", mod._NAME_PATTERN))  # author names never in a deposit
    return named


_LEAK_PATTERNS = _load_leak_patterns()   # list of (name, compiled-regex)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hashed(value) -> dict:
    return {"sha256": sha256_hex(json.dumps(value, sort_keys=True))}


def redact(obj):
    """Recursively replace host/internal-identifier strings in aggregate JSON."""
    if isinstance(obj, str):
        s = obj
        for _name, pat in _LEAK_PATTERNS:
            s = pat.sub("[redacted]", s)
        return s
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def _fail(msg: str):
    print(f"build_deposit: ABORT - {msg}", file=sys.stderr)
    raise SystemExit(2)


# --- RULER ------------------------------------------------------------------
_RULER_AGG_NAMES = {"combined_summary.json", "matrix.json", "run_config.json",
                    "manifest.sha256"}


_DIGEST_FIELDS = {"item_sha256", "expected_answer_sha256", "prompt_sha256", "input_sha256"}
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def clean_ruler_record(rec: dict) -> dict:
    if not isinstance(rec, dict):
        _fail("non-dict item record")
    out = {k: v for k, v in rec.items() if k in _RULER_ITEM_ALLOW and k not in _PROMPTISH}
    for field in _DIGEST_FIELDS:
        if field in out and not (isinstance(out[field], str) and _DIGEST_RE.fullmatch(out[field])):
            _fail(f"{field} is not a lowercase sha256 hex digest; refusing to ship it")
    if rec.get("task") in _QA_TASKS:
        for f in _QA_HASH_FIELDS:
            if out.get(f) is not None:
                out[f] = _hashed(out[f])
    return out


def _is_ruler_item(data) -> bool:
    return isinstance(data, dict) and "task" in data and any(
        k in data for k in ("score", "needlepath_score", "baseline_score")
    )


def build_ruler(src: Path, out_run: Path) -> int:
    n = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if _excluded_path(rel):
            continue
        if f.suffix == ".jsonl":
            dst = out_run / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lines.append(json.dumps(clean_ruler_record(json.loads(line)), sort_keys=True))
                    n += 1
            dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif f.suffix == ".json" and f.name not in _RULER_AGG_NAMES \
                and "summary" not in f.name.lower() and "manifest" not in f.name.lower():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if _is_ruler_item(data):
                dst = out_run / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(clean_ruler_record(data), sort_keys=True), encoding="utf-8")
                n += 1
    for name in ("combined_summary.json", "matrix.json"):
        p = src / name
        if p.exists():
            data = redact(json.loads(p.read_text(encoding="utf-8")))
            (out_run / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
    p = src / "run_config.json"
    if p.exists():
        cfg = redact({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k in _CONFIG_ALLOW})
        (out_run / "run_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return n


# --- RULER gate regression --------------------------------------------------
def build_gate_ruler(src: Path, out_run: Path) -> int:
    n = 0
    gr = src / "gated_rows.json"
    if not gr.exists():
        _fail("gate_ruler: gated_rows.json not found")
    rows = json.loads(gr.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("rows", [])
    cleaned = []
    for r in rows:
        cleaned.append({k: v for k, v in r.items() if k in _GATE_RULER_ALLOW})
        n += 1
    (out_run / "gated_rows.json").write_text(json.dumps(cleaned, indent=0), encoding="utf-8")
    # ship the identity-derived engage/no-regression rollup if present and clean
    for name in ("identity_derived_regression.json", "spot_check_summary.json"):
        p = src / name
        if p.exists():
            (out_run / name).write_text(
                json.dumps(redact(json.loads(p.read_text(encoding="utf-8"))), indent=1),
                encoding="utf-8")
    return n


# --- AppWorld ---------------------------------------------------------------
def _appworld_agg(obj):
    """Keep numeric aggregate structure; drop task-id-bearing keys and text."""
    if isinstance(obj, dict):
        return {k: _appworld_agg(v) for k, v in obj.items() if k not in _APPWORLD_DENY_KEYS}
    if isinstance(obj, list):
        return [_appworld_agg(v) for v in obj if isinstance(v, (int, float, bool, dict))]
    if isinstance(obj, str):
        return redact(obj)
    return obj


def _clean_episode(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in _APPWORLD_EPISODE_ALLOW or _APPWORLD_EPISODE_NUM_RE.search(k):
            if isinstance(v, (int, float, bool)) or k in ("arm", "split", "condition", "termination_reason"):
                out[k] = redact(v) if isinstance(v, str) else v
        elif k == "task_id":
            out["task_id"] = _hashed(v)
    return out


def build_appworld(src: Path, out_run: Path) -> int:
    n = 0
    # 1. numeric aggregate rollups (summaries, official scores, gate analysis)
    for p in sorted(src.rglob("*.json")):
        rel = p.relative_to(src)
        low = p.name.lower()
        if _EXCLUDE_PATH.search(str(rel)):
            continue
        if any(t in low for t in ("summary", "official_scores", "analysis", "paired")):
            dst = out_run / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(_appworld_agg(json.loads(p.read_text(encoding="utf-8"))), indent=1),
                           encoding="utf-8")
            n += 1
    # 2. per-episode metrics (numeric; task_id hashed), from metrics/*.jsonl
    for p in sorted(src.rglob("*_episodes.jsonl")):
        rel = p.relative_to(src)
        if _EXCLUDE_PATH.search(str(rel)):
            continue
        dst = out_run / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(json.dumps(_clean_episode(json.loads(line)), sort_keys=True))
                n += 1
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 3. run_config (allowlisted; the command field with paths/s3 is dropped)
    p = src / "run_config.json"
    if p.exists():
        cfg = redact({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                      if k in _APPWORLD_CONFIG_ALLOW})
        (out_run / "run_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return n


# --- controls ---------------------------------------------------------------
def _clean_control(rec: dict, hash_text: bool) -> dict:
    out = {k: v for k, v in rec.items() if k in _CONTROLS_ALLOW}
    for k in _CONTROLS_TEXT:
        if rec.get(k) is not None:
            out[k] = _hashed(rec[k]) if hash_text else rec[k]
    return out


def build_controls(src: Path, out_run: Path, hash_text: bool) -> int:
    n = 0
    for f in sorted(src.rglob("*.json")):
        rel = f.relative_to(src)
        low = f.name.lower()
        if _EXCLUDE_PATH.search(str(rel)) or "summary" in low or "card" in low or "ledger" in low:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        recs = data if isinstance(data, list) else [data]
        if not (recs and isinstance(recs[0], dict) and "arm" in recs[0]):
            continue
        dst = out_run / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cleaned = [_clean_control(r, hash_text) for r in recs]
        dst.write_text(json.dumps(cleaned if isinstance(data, list) else cleaned[0], sort_keys=True),
                       encoding="utf-8")
        n += len(cleaned)
    return n


# --- GSM8K hash-only fallback ----------------------------------------------
_GSM8K_ROW_ALLOW = {"method", "arm", "budget_label", "condition", "n", "em_rate",
                    "token_reduction_pct"}


def build_gsm8k_hash(src: Path, out_run: Path) -> int:
    """No per-item controls run is recoverable; ship the full-set aggregate,
    a SHA-256 manifest, and a regenerate-via-fetch note (see NOTICE / DEPOSIT.md)."""
    agg = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else \
        json.loads((src / "aggregate.json").read_text(encoding="utf-8"))
    rows = agg if isinstance(agg, list) else agg.get("rows", [])
    cleaned = [{k: v for k, v in r.items() if k in _GSM8K_ROW_ALLOW} for r in rows]
    (out_run / "aggregate.json").write_text(json.dumps(cleaned, indent=1), encoding="utf-8")
    (out_run / "README.md").write_text(
        "# GSM8K control (hash-only)\n\n"
        "This GSM8K figure is from a separate full-set run over all 1319 test items; it was "
        "NOT produced by the public `controls_v1` harness, and no per-item `controls_v1` GSM8K "
        "run exists. Only the full-set exact-match aggregate (method x budget x condition, "
        "n=1319) and a SHA-256 manifest are deposited. Per-budget needlepath-minus-full "
        "exact-match deltas range from -0.15 to +0.38 percentage points. GSM8K is MIT-licensed; "
        "regenerate per-item inputs from the upstream dataset at the pinned revision recorded in "
        "`DATASETS.md` and re-run the `csbench` gsm suite to reproduce the aggregate.\n",
        encoding="utf-8")
    return len(cleaned)


# --- manifest + scrub -------------------------------------------------------
def write_manifest(out_run: Path) -> None:
    lines = []
    for f in sorted(out_run.rglob("*")):
        if f.is_file() and f.name != "manifest.sha256" and ".git" not in f.parts:
            digest = sha256_hex(f.read_text(encoding="utf-8", errors="surrogatepass"))
            lines.append(f"{digest}  {f.relative_to(out_run)}")
    (out_run / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scrub(out_dir: Path) -> None:
    """Scan the assembled deposit for sensitive LITERALS (see _load_leak_patterns).
    Independent of the repo's prose-hygiene gate so benign third-party text does not
    false-positive; fails the build on any hit."""
    hits = []
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file() or ".git" in f.parts:
            continue
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        for lineno, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            for name, pat in _LEAK_PATTERNS:
                if pat.search(line):
                    hits.append((f.relative_to(out_dir), lineno, name, line.strip()[:120]))
    for rel, lineno, name, snip in hits[:25]:
        print(f"LEAK {rel}:{lineno}  [{name}]  {snip}")
    if hits:
        _fail(f"deposit leak-scan found {len(hits)} hit(s)")
    print("deposit leak-scan clean: 0 hits")


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble a license-gated public deposit for a run.")
    ap.add_argument("--run-type", required=True,
                    choices=["ruler", "gate_ruler", "appworld", "controls_plain",
                             "controls_hash", "gsm8k_hash"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="deposit")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    out_run = out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=True)

    if args.run_type == "ruler":
        n = build_ruler(src, out_run)
    elif args.run_type == "gate_ruler":
        n = build_gate_ruler(src, out_run)
    elif args.run_type == "appworld":
        n = build_appworld(src, out_run)
    elif args.run_type == "controls_plain":
        n = build_controls(src, out_run, hash_text=False)
    elif args.run_type == "controls_hash":
        n = build_controls(src, out_run, hash_text=True)
    elif args.run_type == "gsm8k_hash":
        n = build_gsm8k_hash(src, out_run)
    else:  # pragma: no cover
        _fail(f"unknown run-type {args.run_type!r}")

    write_manifest(out_run)
    scrub(out_root)
    if n == 0:
        _fail(f"{args.run_id}: no per-item records were assembled from {args.src} for run type {args.run_type}; nothing to deposit")
    print(f"build_deposit: {args.run_id} OK - {n} records, manifest + scrub clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
