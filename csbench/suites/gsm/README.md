# GSM boundary suite (GSM8K + GSM-IC)

**This suite is a DIAGNOSTIC, not a flagship result.** It exists to disclose two
*boundary* cases for context selection honestly (cases where record-level
selection is expected to add little or nothing), not to headline a win. The
flagship claims for context selection live with the RULER long-context suite and
the tool / document QA suites, **not here**. No headline or marketing numbers
appear anywhere in this suite by design.

## The two disclosed boundary cases

1. **GSM8K: selection is a no-op by design.** GSM8K problems are clean,
   self-contained, single-document math word problems. The whole problem
   statement is relevant, so there is nothing to select away. A correct selector
   preserves the whole prompt; the expected, correct outcome is that a selection
   arm matches the full-context baseline. That is not a null finding: it is the
   right behaviour for a regime with no irrelevant content to remove.

2. **GSM-IC: the distractor is outside the selection regime.** GSM-IC injects
   exactly one irrelevant sentence *inline*, inside the single question record a
   solver must keep. Because the distractor lives *within* one record, it is
   outside the record-level selection regime this benchmark studies: a selector
   that operates on whole records cannot drop a distractor sentence without
   dropping the one record it has to preserve. GSM-IC is included precisely to
   make that boundary explicit and auditable.

Reporting both boundaries plainly is the point of the suite. Any comparison it
produces should be read against these two framings.

## What the runner does

`runner.py` runs a matched-token-budget sweep: every arm is driven to the **same**
reduction targets (25% / 50% / 75%, plus each arm's native default) and compared
at equal budget. Budgets are resolved as a percentage of the **question**
content's token count, never the full prompt's: the fixed few-shot block is
always-present scaffolding that no method reduces, so the budget label constrains
exactly the selectable content, and every method runs on the identical few-shot
scaffolding (no method gets a setup edge).

Per `(arm, budget, condition, item)` the driver:

1. resolves the per-item token budget (`resolve_budget_tokens`);
2. builds a per-arm `ContextRequest` (`csbench.suites.gsm.request.build_gsm_request`);
3. runs the uniform arm contract `arm.select(request) -> ContextResponse`;
4. reassembles the model prompt (`assemble_model_prompt`): the `full_context`
   arm's rendered context is byte-identical to the raw prompt and is sent
   verbatim; every other arm's rendered context is the reduced question content,
   so the identical few-shot block is re-attached verbatim;
5. calls the model via `csbench.openai_client.call_openai` (temperature 0);
6. scores GSM8K exact-match via `csbench.stats.score_exact_match` against the
   known gold numeric answer.

It aggregates per `(arm × budget × condition)` cell: exact-match rate, token
reduction, fallback rate, latency, and (for GSM-IC only) a per-**base-problem**
`macro_accuracy` (a base problem counts as correct only if every sampled variant
of it is individually correct). Macro accuracy has a small true N and a wide
confidence interval; it is a coarse secondary sanity check and is **never**
headlined over the micro exact-match rate.

## Data

- **GSM8K**: real GSM8K via the Hugging Face `datasets` hub (the `test` split
  for evaluation items, the `train` split for the fixed few-shot exemplars).
  DISTRACTOR items append a clearly-labelled block of irrelevant filler drawn
  from *other* GSM8K questions.
- **GSM-IC**: the real, public `GSM-IC_2step.json` file, sampled uniformly per
  base problem (a balanced stratified sample). DISTRACTOR items use the injected
  sentence already inline in the question; CLEAN items use the original question.

`datasets` and `openai` are **optional** dependencies, imported lazily inside the
loaders / model client, so the modules import without them.

## Running

```bash
# a running-spend hard stop is mandatory for any real run
export BENCH_BUDGET_USD=25
export OPENAI_API_KEY=...

# GSM8K (clean single-document math; selection is a no-op by design)
python -m csbench.suites.gsm.runner \
  --mode gsm8k \
  --arms full_context,needlepath,llmlingua2 \
  --needlepath-url https://<hosted-endpoint> --operating-point np-2026-07-r1 \
  --n 200 --out runs/gsm8k

# GSM-IC (inline distractor; outside the record-level selection regime)
python -m csbench.suites.gsm.runner \
  --mode gsm_ic \
  --arms full_context,needlepath,llmlingua2 \
  --needlepath-url https://<hosted-endpoint> --operating-point np-2026-07-r1 \
  --n 2400 --out runs/gsm_ic
```

The runner **prints the cost estimate first and aborts before any API call** if
the estimated spend would exceed `BENCH_BUDGET_USD` (the estimate comes from
`csbench.suites.gsm.cost`, over real data and the same per-item budget math the
sweep uses, so the printed figure describes the run that will actually happen).

### Flags

| Flag | Meaning |
| --- | --- |
| `--mode` | `gsm8k` or `gsm_ic` |
| `--arms` | comma-separated: `full_context,needlepath,llmlingua2,cpc` |
| `--n` | item count (GSM8K: first-n `test` items; GSM-IC: stratified sample size, a multiple of the base-problem count, otherwise a plain first-n smoke slice) |
| `--budgets` | comma-separated budget labels; default per mode (`gsm8k`: `25pct,50pct,75pct,native_default`; `gsm_ic`: `25pct,75pct,native_default`, no 50%) |
| `--model` | answer model (default `gpt-4o-mini`; see `cost.py` for the pricing caveat) |
| `--needlepath-url` | hosted Needlepath endpoint (required for the `needlepath` arm) |
| `--operating-point` | opaque hosted operating-point label (default `np-2026-07-r1`) |
| `--seed` | selection / distractor seed |
| `--out` | output root |
| `--concurrency` | max in-flight model completions per cell |

Both conditions (`clean`, `distractor`) are always swept. The `needlepath` arm
reads its bearer token from `NEEDLEPATH_API_KEY` in the environment only.

## Outputs (under `--out`)

- `run_config.json`: invocation, knobs, provenance.
- `git_provenance.json`: neutral git snapshot.
- `items/<condition>/<arm>/<budget_label>.jsonl`: one row per item (see schema
  below), written incrementally per cell.
- `summary.json` / `summary.md`: per-cell summary rows.
- `manifest.sha256`: `<sha256>  <relpath>` over every per-item file, verifiable
  with `tools/verify_manifests.py --base-dir <out>`.

### Per-item row schema

Correctness + budget: `mode`, `arm`, `budget_label`, `condition`, `item_id`,
`base_problem_id` (GSM-IC only, else `null`), `gold_numeric`,
`resolved_budget_tokens`, `answer`, `correct`, `model_latency_ms`.

Read straight off `ContextResponse`: `tokens_before`, `tokens_after`,
`tokens_saved`, `reduction_ratio`, `records_available`, `records_selected`,
`fallback_used`, `selection_error`, `engine_latency_ms`, `budget_tokens`,
`attempted_budget_tokens`, `selected_record_ids`.

Neutral safety subset (`null` for pure compression / full-context arms):
`selection_safe`, `fallback_reason`, `coverage_score`, `evidence_shape`.

### Per-cell summary schema

`mode`, `arm`, `budget_label`, `condition`, `n`, `em_rate`, `macro_accuracy`
(GSM-IC only), `n_base_problems`, `failed_base_problem_ids`,
`avg_reduction_ratio`, `fallback_rate`, `mean_engine_latency_ms`,
`mean_model_latency_ms`, `n_prep_errors`, `n_completion_errors`.
