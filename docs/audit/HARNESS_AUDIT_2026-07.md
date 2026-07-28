# Harness audit, July 2026

Adversarial cross-model audit of the context-selection-bench harness.

| | |
| --- | --- |
| Date | 2026-07-25 |
| Scope A | released harness at v1.0.1, `main` @ `50ace32` (`csbench/`, `tools/`, `scripts/`) |
| Scope B | unmerged suite adapters, `docfinqa` @ `95c2741` (superset of `financebench` @ `62ed88c`) |
| Reviewers | gpt-5.5 (max thinking effort, statistics lead) and gemini-3.1-pro-preview (max thinking effort), independently, then cross-refuting |
| Orchestrator | Claude Opus 5, acting as skeptic of both reviewers |
| Status | read-only. No code was changed. No published number or artifact was touched. |

Sol (gpt-5.6) was not available in this environment; the OpenAI seat was filled by
gpt-5.5, the current top-of-line model available here, with the user's approval.

## Method

Each reviewer audited the code independently with no sight of the other's output,
across five sections: protocol fidelity, metric permissiveness, silent-misuse
surfaces, reproducibility integrity, and statistics. Each then received the other's
findings and attempted to refute them, with `REFUTED` / `OVERSTATED` / `CONFIRMED`
verdicts. The orchestrator independently verified every line citation and every
claim of impact against the source and against the published run artifacts, and
made the final severity call. Findings that survived only as assertions, without a
demonstrated path to a wrong number, were downgraded or dropped.

Three defects already known in the scope B adapters (DocFinQA answer-protocol
mismatch, four-word-fragment evidence locator, question-keyword metadata pollution)
were excluded from both passes by instruction. This audit hunts the same classes
elsewhere.

Line numbers below were each re-verified by the orchestrator against the stated
commits; several reviewer citations were off by 10-60 lines and have been corrected.

---

## Summary

> **AMENDED 2026-07-25 after recomputation from the deposited per-item rows.**
> Findings 1, 2 and 5 below were materially wrong and are corrected in place; a new
> finding 1b replaces the original finding 1 as the defect that actually moves
> published numbers. A claim made during the recomputation and since disproven is
> recorded in "Disproven claims" at the end. Full working:
> `scratchpad/RECOMPUTATION_ruler_repl_v1.md`.

| # | Finding | Severity | Published v1.0.1? |
| --- | --- | --- | --- |
| 1 | ~~RULER paired statistics computed on a non-unique `index` join~~ **CORRECTED: mechanism wrong, join is harmless** | ~~Critical~~ Informational | **No** |
| 1b | RULER cell accuracy/EM/N averaged over duplicated rows | Critical | **Yes** |
| 2 | RULER resume path keyed by non-unique `index` writes duplicate rows | Critical (root cause) | **Yes, confirmed fired** |
| 3 | Holm correction named as the primary test, not implemented anywhere | High | **Yes (disclosure)** |
| 4 | Output manifests omit `matrix.json` and every summary artifact | High | **Yes** |
| 5 | BFCL: zero expected values returns an unconditional pass | High | **No (latent, corrected)** |
| 6 | BFCL: unanchored substring matching on argument values | High | **Yes** |
| 7 | SQuAD: raw substring containment can flip preservation to true | High | **Yes** |
| 8 | `csbench/suites/ruler/stats.py` cannot read the harness it ships with | High | No (dead path) |
| 9 | RULER `--n 0` silently means "load the entire file" | Medium | No |
| 10 | SQuAD and BFCL dataset revisions unpinned | Medium | **Yes** |
| 11 | GSM: failed items leave the denominator after their outcome is known | Medium | Diagnostic only |
| 12 | QA runner records a `--seed` it never uses | Medium | **Yes (config)** |
| 13 | Empty result sets summarize to a confident `0.0` | Medium | No |
| 14 | TruthfulQA zero-flips is structural, reported as empirical | Medium | **Yes (framing)** |
| B1-B10 | Scope B adapter findings | see below | No |

Findings 1b, 2, 3 and 4 are supersession decisions, not patches. They are flagged for
your judgment and no action has been taken.

---

# Scope A: released harness v1.0.1

## 1. CORRECTED: the `index` join is harmless, not the defect

**Original severity: Critical. Corrected severity: Informational. Does NOT affect any
published statistic.**

**What this section originally claimed, and why it was wrong.** The audit asserted that
the `index` join at `harness.py:536` silently discarded items and corrupted the paired
statistics. Recomputation from the deposited per-item rows disproves the mechanism.

The duplicate rows in the cell files are **byte-identical copies of the same item**
(7800 rows, 7404 distinct `(position, index)`, zero non-identical duplicate groups, zero
disagreements on `correct`). The `index` join therefore *deduplicates* and recovers
exactly the correct distinct-item population. Recomputing all 52 arm-vs-baseline
comparisons under both the published index join and correct position pairing gives
**numerically identical results in 52 of 52 cases, with zero significance flips.**

The `n_paired < 100` values are not evidence of loss; they are the true distinct-item
counts. The number that is wrong is the `N=100` reported beside them.

The real defect is the resume path that created the duplicates (finding 2) and the
accuracy aggregation that consumes them (finding 1b). The original text of this finding
is superseded.

**Retained sub-point.** The join is still the wrong key on principle, and Appendix E's
description ("paired by row position ... with a per-position alignment assertion") is
still inaccurate for `harness.py`, which performs no such assertion. Had the duplicates
differed, this would have been the critical finding it was described as. Fix remains
worthwhile as defence in depth; it changes no published number.

## 1b. RULER cell accuracy, EM and N are averaged over duplicated rows

**Severity: Critical. Affects published v1.0.1 numbers.**

`csbench/suites/ruler/harness.py:500-511` (`summarize_cell`)

```python
n = len(rows)
"accuracy":    _mean(float(r.get("score", 0.0)) for r in rows),
"em_accuracy": _mean(1.0 if r.get("correct") else 0.0 for r in rows),
```

`rows` is the full cell file including the duplicate rows written by finding 2. An item
whose row was emitted three times is counted three times. Every RULER cell accuracy, EM,
`N`, reduction, latency and token average in `matrix.json`, `matrix.md` and the
`combined_summary.json` rollups is a mean over a pseudo-replicated population.

**Confirmed against the published artifact.** Across the 78 cell files, 7800 rows reduce
to 7404 distinct `(position, index)` pairs. 42 of 78 cells shift when recomputed over
distinct items. Mean absolute shift 0.0039; maximum 0.0288:

```
16k niah_single_2   full_context  N=100 true=86  acc 0.320 -> 0.349  (+0.029)
16k niah_single_2   compresr      N=100 true=86  acc 0.220 -> 0.244  (+0.024)
16k niah_multikey_3 needlepath    N=100 true=82  acc 0.200 -> 0.183  (-0.017)
8k  niah_multikey_3 needlepath    N=100 true=67  acc 0.120 -> 0.134  (+0.014)
```

Every cell reports `N=100`; the true distinct-item count ranges 67 to 99.

**Effect on the replication table.** The per-length and pooled figures reproduce exactly
from the raw duplicate-retaining rows (n=1300 per length, n=2600 pooled). Recomputed
over distinct items:

| Scope | n pub -> corr | delta pub -> corr | p pub -> corr |
| --- | --- | --- | --- |
| NP 8K | 1300 -> 1231 | +4.64 -> +4.48pp | 0.0144 -> 0.0305 |
| NP 16K | 1300 -> 1237 | +5.33 -> +4.77pp | 0.0031 -> 0.0221 |
| NP pooled | 2600 -> 2468 | +4.99 -> +4.62pp | 0.0001 -> 0.0013 |
| Compresr pooled | 2600 -> 2468 | +1.45 -> +1.58pp | 0.1925 -> 0.1662 |

**No claim loses significance.** Under Holm over the co-primary per-length family, NP
remains significant at both lengths (corrected smallest p 0.0221 against a 0.025
threshold, clearing by 0.0029) and Compresr remains non-significant at both. Direction
is unchanged throughout. On the evidence this is a correction, not a retraction; that
ruling is the maintainer's.

**Not applicable to `ruler_v1`.** That deposit stores one file per position with all
arms co-located, so no join occurs and no duplicates exist (0 duplicate positions across
26 task-directories). All `ruler_v1`-backed claims stand.

**Proposed fix.** Deduplicate on `(position, index)` at load, requiring byte-identical
copies and failing otherwise; report `n_distinct` beside `n_rows`. Do not silently
re-derive published numbers: any restatement is a supersession decision.


## 2. RULER resume path is keyed by the same non-unique `index`

**Severity: Critical. This is the root cause. It demonstrably fired in the published
run.**

`csbench/suites/ruler/harness.py:426` and `:466`

```python
rows[int(row["index"])] = row          # 426, in _load_existing_rows
...
prior = existing.get(example.index)    # 466, in run_task
```

The same non-unique key is used for the resume cache. On resume, every position
whose example carries a duplicated `index` reads back the *same* cached row, which
is then re-stamped with the current `position` at line 469. One item's model output
is copied onto several positions, and the copies are indistinguishable from real
results downstream.

**Failure scenario.** A run of `niah_multikey_3` is interrupted and resumed. Of the
100 positions, 33 share indices. Those positions load a prior row belonging to a
different item, get relabelled with the new position, and are written to the cell
JSONL as if independently evaluated. Cell accuracy is then computed over duplicated
outcomes.

**Confirmed fired, and this resolves the question the audit originally left open.**
The deposited cell files contain 7800 rows but only 7404 distinct `(position, index)`
pairs; 42 of 78 files carry duplicates. Because `prior` is a shared dict that is mutated
and stored under several positions, the emitted copies are byte-identical, which is
exactly the pattern observed (zero non-identical duplicate groups). The published
`ruler_repl_v1` run therefore did resume. `run_config.json` records no resume flag, so
this was invisible in the artifact until recomputation.

Cross-confirmed against `ruler_v1`, which never resumed through this path: 8k
`niah_multikey_3` has exactly 33 duplicate RULER indices, giving 67 distinct, matching
`ruler_repl_v1`'s `n_paired=67` for that cell precisely.

**Proposed fix.** Key the resume cache on `position`, and cross-check the stored
`index` matches the example's before reusing a row. Record whether a run resumed in
`run_config.json`.

Found by gpt-5.5 during the refutation pass, not in either first pass. Verified by
the orchestrator.

## 2b. RECURRENCE: index-based pairing was found and fixed once already

**Severity: Critical (process). This is the same defect class re-entering through a
second code path, not a new incident.**

**First occurrence.** `docs/needlepath/RULER_RESULTS.md:95` (nextmoca), under
"A real bug was found and fixed while running it against real data":

> the module aligned rows by RULER's own `index` field, assuming (per its own docstring)
> that "row alignment is trivially guaranteed by construction." That assumption was true
> for the synthetic fixtures but **false for the real RULER dataset** - many tasks
> (`niah_single_1/2/3`, `niah_multikey_1/3`, `niah_multivalue`, `niah_multiquery`) have
> genuine duplicate `index` values ... **Fix:** `ruler_stats.py` now aligns rows by
> `position` instead of `index`.

**Second occurrence.** `csbench/suites/ruler/harness.py` re-entered the identical trap on
two independent lines: `:426`/`:466` (resume cache keyed by `index`, findings 2 and 1b)
and `:536` (comparison join keyed by `index`, finding 1). The task list in the original
bug record is the same task list that shows duplicates in `ruler_repl_v1`.

**Why the first fix did not hold.** It was local. `ruler_stats.py` was corrected and
documented, but the correction lived in one module rather than in a shared utility, so a
second code path written later re-derived the same wrong assumption from scratch. The
public repository now ships both the fixed module and the unfixed path side by side.

**Structural fix required. No third code path should be able to reintroduce this.**

1. **One shared pairing utility.** A single position-based pairing function that every
   path must call. `harness.compare_to_baseline`, the resume cache, and the proposed
   aggregation tool all route through it. Delete the ad-hoc dict joins rather than
   repairing them, so there is no second implementation to drift.
2. **Duplicate-index tripwire.** An assertion that fires wherever rows are joined or
   persisted: if `index` is used as a key anywhere, fail loudly. Cheap, and it converts
   the silent failure mode into a startup error.
3. **Item-identity test.** Assert every persisted row set has unique `(position, index)`
   and that the row count equals the requested `n`. This fails today against the
   published `ruler_repl_v1` artifacts, which is exactly the point: it is the regression
   test that would have caught this at write time rather than at audit time.

Per standing supersession discipline, the fix ships together with test 3.

## 3. Holm correction is named as the primary decision rule and implemented nowhere

**Severity: High. Affects published disclosure.**

Appendix E states the primary test is "a pair of co-primary per-length McNemar
tests ... Holm-corrected within the two-test family at alpha=0.05".

A case-insensitive search for `holm` across `csbench/`, `tools/`, `scripts/` and
`tests/` returns nothing. The only occurrence anywhere in the repository is in
`paper/main.tex`. `csbench/stats.py:mcnemar_test` returns a raw p-value;
`harness.py:564` writes it to `matrix.json` as `mcnemar_p_value`; `matrix.md`
renders it unadjusted.

**Failure scenario.** A JOSS reviewer or replicator runs the released harness,
obtains a per-cell p of 0.04, and reads it as significant at the paper's stated
alpha. Under the disclosed Holm procedure over the two-test family it would be
compared against 0.025 and would not be significant. Nothing in the harness output
signals that the emitted p-values are raw.

**Mitigating check.** The published significant results are p = 0.0075, 0.0007,
0.0039, 0.0013, 0.0013 and 0.0001, all comfortably below 0.025, so applying Holm by
hand would not flip any of them *as computed*. That mitigation is contingent on
finding 1, because those p-values were computed on the wrong population to begin
with.

The two reviewers split on severity (critical vs medium). Adjudicated as High: the
released code does not implement the paper's stated primary test, which is a
reproducibility defect independent of whether the conclusions survive.

**Proposed fix.** Implement Holm over the declared family and emit both raw and
adjusted p-values plus explicit family membership. If the correction was applied
outside this code, the harness output must state that its p-values are raw.

## 4. Output manifests omit `matrix.json` and every summary artifact

**Severity: High. Affects the published artifact's integrity guarantee.**

`csbench/suites/ruler/harness.py:812` initialises `output_files`, and only per-cell
JSONL paths are appended (`:841`). `write_matrix` and `write_combined_summary` run
at `:856-857`, and `run_config.json` and `trusted_benchmark_manifest.json` are
written at `:796` and `:808`. None are added before
`write_output_manifest(...)` at `:858`. The same shape appears at
`csbench/suites/qa_runner.py:572`/`:612` and `csbench/suites/gsm/runner.py:873`.

**Confirmed against the published artifact.** `runs/ruler_repl_v1/manifest.sha256`
lists exactly 78 paths, all `items/**/*.jsonl`. The directory also contains
`matrix.json`, `matrix.md`, `combined_summary.json`, `combined_summary.md`,
`run_config.json` and `trusted_benchmark_manifest.json` - none covered.

**Failure scenario.** Any headline number in `matrix.json` is edited after the run.
`tools/verify_manifests.py` reports success, because the file it would need to hash
is not in the manifest. The per-item rows that *are* covered would contradict the
edited matrix, but nothing in the tooling compares them.

**Proposed fix.** Append every generated artifact except `manifest.sha256` itself to
`output_files` before writing the manifest, in all three runners.

Both reviewers found this independently; both confirmed it. gpt-5.5 correctly noted
that the QA runner's glob does cover per-suite `rows.json` and `summary.json`, so
the gap there is narrower than stated, but the root-level config and summary are
still uncovered.

## 5. BFCL scores an unconditional pass when a call has no expected values

**Severity: High as a latent defect. CORRECTED: does NOT affect published v1.0.1
numbers.** The audit originally marked this "affects published: Yes". That was wrong.
Running the scorer's `values_total` logic over all 600 deposited BFCL artifacts
(`deposit/controls_v1/bfcl/`, 100 distinct cases) shows the free-pass branch fires on
**0 of 100 cases**. Every published BFCL case has at least one checkable value. The
defect is real and should be fixed before the suite is reused or extended, but no
published number rests on it.

`csbench/suites/bfcl.py:210`

```python
if values_total == 0:
    return True
```

If every accepted value for a function call is empty or `None`, `values_total`
never increments and the item is credited as correct without inspecting the model
response at all. The response could be empty, a refusal, or unrelated text.

This is a denominator-of-zero crediting path: the exact shape of the calibration
defects. Neither reviewer flagged it; found by the orchestrator while verifying
reviewer citations around `bfcl.py:194-212`.

**Failure scenario.** A BFCL `simple` case whose ground truth has only optional
parameters with empty defaults. Every arm scores 1.0 on it regardless of output.
The item contributes to the accuracy numerator and denominator for all arms
identically, which inflates absolute accuracy and dilutes any real difference
between arms toward the null - flattering a "no detectable difference" conclusion.

**Proposed fix.** Return `None` / not-scorable for calls with no checkable values
and exclude them from the denominator with an explicit count, rather than crediting
them.

## 6. BFCL argument matching is unanchored substring matching

**Severity: High. Affects published v1.0.1 numbers.**

`csbench/suites/bfcl.py:194` (`if value_text in response_lower`) and `:208`
(numeric variants, same unanchored test).

Expected value `12` matches inside `1129`. Expected `1` matches inside `10`, `100`,
or any date. The plain-string test at `:194` fires before the numeric-variant path,
so it applies to every value type.

Combined with the `>= 0.5` rule at `:212`, a call with two expected values needs
only one accidental substring collision to be scored as matched.

**Appendix disclosure: partial.** Appendix B discloses the proxy and the
"at least half of the expected parameter values appear in the response" rule, and
calls it the largest deviation. It does not disclose that "appear in" is an
unanchored substring test with no word boundaries and no parameter-name binding.
A reader would reasonably assume value-level matching.

Both reviewers found this. gemini rated it critical, gpt-5.5 argued overstated on
the grounds that Appendix B discloses the proxy and claims no absolute BFCL score.
Adjudicated as High: the disclosure covers the proxy's design but not this
matching semantics, and the calibration failure is real.

**Proposed fix.** Require word boundaries for numeric and alphanumeric values, and
bind values to their parameter names. Report the count of matches that relied on a
bare substring hit.

## 7. SQuAD containment check can flip preservation to true on a raw substring

**Severity: High. Affects published v1.0.1 numbers.**

`csbench/suites/squad.py:255-259`

```python
contains_gt = (
    case.ground_truth.lower() in response_compressed.lower()
    or compute_f1(response_compressed, case.ground_truth) > _CONTAINS_GT_F1_THRESHOLD
)
preserved = f1 > _F1_PRESERVED_THRESHOLD or contains_gt is True
```

`preserved` is an OR. A raw, untokenized, unanchored substring hit on the gold
answer is sufficient on its own, regardless of the full-vs-reduced F1 comparison
that is supposed to be the actual metric.

**Failure scenario.** Gold is `car`. The full-context answer is `vehicle`, the
reduced-context answer is `Oscar`. `compute_f1("vehicle", "Oscar") = 0`, so the
preservation test fails, but `"car" in "oscar"` is `True`, so `contains_gt` is
`True` and the item is recorded as preserved. Short golds (`1`, `a`, `US`) match
almost anything.

**Appendix disclosure: partial.** Appendix B discloses that the gold answer is used
"only as a secondary containment check". It does not disclose that this check is an
unanchored substring match, nor that it can single-handedly set the reported
preservation outcome via the OR.

Both reviewers found this independently and both confirmed it.

**Proposed fix.** Use SQuAD's own normalized token matching for containment, and
report how many preserved outcomes rested on containment alone.

## 8. `csbench/suites/ruler/stats.py` cannot read the harness it ships with

**Severity: High. No published number depends on it.**

`csbench/suites/ruler/stats.py:180` builds
`output_root / f"{run_name}_{length_label}" / task / "rows.json"`.
`csbench/suites/ruler/harness.py:410` writes
`out_root / "items" / length / arm_name / f"{task}.jsonl"`.

These layouts are incompatible. The module that implements the correct
position-based pairing and the identity assertions described in Appendix E cannot
consume the current harness's output at all.

This matters more than a dead code path normally would, for two reasons. The
repository ships two RULER statistics implementations that disagree with each other
on the load-bearing question of how to pair items (finding 1), and this is the one
whose behavior the paper describes. And a reviewer directed to the statistics module
will read correct code that never runs.

**Failure scenario.** A replicator runs the harness, then points `ruler/stats.py` at
the same `output_root`. It finds no `rows.json` and produces nothing. If a stale
`rows.json` from an older harness version is present, it silently computes
statistics over the wrong run.

**Proposed fix.** Either update the module to read the current per-arm JSONL layout
and make the harness call it, or quarantine it with an explicit note. Shipping two
disagreeing statistics paths under JOSS review should be resolved before merge.

## 9. RULER `--n 0` silently means "load the entire file"

**Severity: Medium. Not used for published numbers.**

`csbench/suites/ruler/harness.py:892` declares `--n` with no lower-bound validation.
`csbench/suites/ruler/data.py:145` reads `if limit and len(examples) >= limit:`.
With `limit=0` the guard is falsy and the break never fires, so the loader returns
every row in `test.jsonl`. `run_config.json` records `n: 0`.

`csbench/suites/qa_runner.py:678` has the guard (`if args.n < 1`) that the RULER
harness lacks, which shows the intended contract.

**Failure scenario.** A reviewer runs `--n 0` expecting a no-op smoke test, and
instead starts a full-file run against a paid API, recorded in the config as `n=0`.

**Proposed fix.** Reject `--n < 1` in the RULER CLI, matching the QA runner, and
change the loader guard to `if limit is not None and ...`.

## 10. SQuAD and BFCL dataset revisions are unpinned

**Severity: Medium. Affects published reproducibility.**

`csbench/suites/squad.py:46`: `SQUAD_REVISION: str | None = None`
`csbench/suites/bfcl.py:54`: `BFCL_DATASET_REVISION = "main"`

Both resolve to moving targets. `bfcl.py:159` records the symbolic string `"main"`
into the run artifact, which proves nothing about the bytes fetched. Appendix B
states datasets are obtained from official sources but does not pin snapshots for
these two.

**Failure scenario.** A BFCL maintainer updates `BFCL_v3_simple.json`. A replicator
runs a month after publication, gets different scores, and has no artifact-level way
to distinguish upstream data drift from a method difference.

**Proposed fix.** Pin both to immutable commit SHAs and record the SHA-256 of every
fetched payload in the run artifact.

## 11. GSM failed items leave the denominator after their outcome is known

**Severity: Medium. Diagnostic control only.**

`csbench/suites/gsm/runner.py:492`: `n = len(item_rows)`, where prep failures
(`:418-430`) and completion failures (`:449-461`) were already excluded.

Because exclusion happens after the failure is observed, and failures correlate with
item difficulty and with how aggressively an arm reduced the context, the surviving
population differs per arm. `macro_accuracy` in `csbench/stats.py` compounds this:
it marks a base problem correct when `all()` of its *surviving* variants are
correct, so losing a variant to a completion error can convert a failed base problem
into a passing one.

**Failure scenario.** A selection arm fails to prepare 20 hard GSM-IC distractor
items and scores 80/80 on the rest, reporting `em_rate = 1.0, n = 80`. A baseline
that attempted all 100 and scored 90 reports 0.90. The arm that failed more looks
better.

Mitigated, not eliminated, by `n_prep_errors` / `n_completion_errors` being recorded
alongside, and by GSM8K being a disclosed diagnostic control whose published result
is neutral. gemini rated this High, gpt-5.5 argued Medium; adjudicated Medium.

**Proposed fix.** Report both `em_rate_scored_only` and an intention-to-treat rate
counting failures as incorrect, and make `macro_accuracy` aware of the expected
variant count per base problem.

## 12. QA runner records a `--seed` it never uses

**Severity: Medium. Affects published config semantics.**

`csbench/suites/qa_runner.py:655` defines `--seed`; `:535` calls
`load_cases(suite, args.n)`, which takes the first `n` cases deterministically. The
seed reaches `run_config.json` and nothing else.

**Failure scenario.** A replicator varies `--seed` to estimate sampling variance,
gets byte-identical item sets every time, and reports a variance of zero. The
published config advertises a reproducibility knob that does not exist.

The same pattern exists at `csbench/suites/ruler/harness.py:898` and
`csbench/suites/gsm/runner.py:924`; in RULER the seed does reach
`bootstrap_ci_delta`, so it is live there and only item selection is unaffected.

**Proposed fix.** Either wire the seed into a seeded sampling mode and record the
selected case ids, or remove it from the QA runner and its config.

## 13. Empty result sets summarize to a confident `0.0`

**Severity: Medium. Not used for published numbers.**

`csbench/suites/qa_common.py:142`:
`"accuracy_preservation_rate": passed / len(results) if results else 0.0`

A run in which every case was filtered out, or every call failed, produces
`cases: 0` next to a full set of `0.0` rates. Rendered into markdown, that reads as
a suite that scored zero rather than a run that did not happen.

**Proposed fix.** Raise on an empty result set for benchmark runs, or emit `null`
and mark the artifact invalid for comparison.

## 14. TruthfulQA's zero-flips result is structural, reported as empirical

**Severity: Medium. Affects published framing, not the number.**

`csbench/suites/qa_runner.py:275-281`. The arm runs (`arm.select(request)`), but the
prompt sent to the model is rebuilt from `case.query` and `case.metadata["choices"]`
and never reads `response.rendered_context`. The arm's output affects only the
token and fallback accounting columns.

The code is honest about this: the function docstring states the prompt is
"byte-identical to `case.context` and independent of the arm's rendered output",
and `csbench/suites/truthfulqa.py:84-88` explains the design rationale.

The finding is about the paper, not the code. Appendix B reports the TruthfulQA
result as "a byte-identical no-op (0.78 on all three arms, zero flipped answers,
100% stand-down), so with identical inputs across arms the metric definition cannot
change the paired result." That attributes the identical inputs to the gate standing
down. In fact the prompt is arm-independent by construction, so zero flips was
guaranteed before any arm ran and regardless of gate behavior. A structural
invariant is presented as an observed control outcome.

gpt-5.5 rated this critical, arguing an arm could delete the context and still score
full accuracy. gemini refuted it as documented intended behavior. Both are partly
right: the code is documented, but the paper's account of *why* the inputs were
identical is wrong. **Downgraded from critical to medium and re-scoped from a code
defect to a disclosure defect.**

**Proposed fix.** Either state in Appendix B that TruthfulQA's prompt is
arm-independent by construction, so it functions as a wiring check rather than as
evidence of preserved quality, or assert
`response.rendered_context == case.context` and report that assertion.

---

# Scope B: unmerged adapters (financebench, docfinqa)

None of these affect any published number. All are pre-merge.

## B1. FinanceBench "Gold text delivered" measures page-prefix delivery

**Severity: High.** `csbench/suites/financebench/suite.py:232`

```python
probe = re.sub(r"\s+", " ", case.document_pages[page_number - 1]).strip()[:probe_chars]
```

The probe is the first 120 characters of the cited *page*, not the annotator's
evidence text. The docstring is accurate ("a hit here means the arm delivered the
very page the annotator cited"); the summary table column is not - it is labelled
"Gold text delivered".

Returns `True` when an arm keeps only a page header and drops the evidence, and
`False` when an arm extracts the exact evidence line without the page prefix. Both
directions are wrong for the label.

**Fix.** Carry `evidence_text` from the dataset into `FbCase` and probe that. Report
page-level delivery separately as `evidence_page_delivered`.

## B2. DocFinQA evidence coverage credits any one gold span

**Severity: High.** `csbench/suites/docfinqa/suite.py:141`

```python
return any(span in haystack for span in locatable)
```

FinQA items routinely need a table row plus surrounding prose. If three spans are
locatable and two are dropped, coverage is still `True`, and the runner's table
labels the column "Evidence coverage". Distinct from the known four-word-fragment
defect; this is the boolean logic over multiple spans.

**Fix.** Report span-level recall and complete-coverage separately, or rename to
`any_gold_span_delivered`.

## B3. DocFinQA sampling contradicts its own prereg

**Severity: High.** `csbench/suites/docfinqa/data.py:170` vs
`docs/prereg/docfinqa_python_execution_v1.md:40-41`

The prereg states the smoke "draws 25 items without replacement from the
identity-sorted population using `random.Random(42).sample`". The code returns
`sorted(cases, key=lambda case: case.sampling_sha256)` - a different hash, derived
from the FinQA `program` and `exe_ans` fields, not `identity_sha256`.

Two implementations of the same frozen protocol, same seed, different 25 items.
A prereg that does not match its implementation was called out in the brief as
high-severity, and this is one.

**Fix.** Sort by `identity_sha256`, or amend the prereg before any scoring run and
record both ordered population hashes in the artifact.

## B4. Neither adapter records the hash of the input data it read

**Severity: High.** `csbench/suites/financebench/runner.py:236-262`,
`csbench/suites/docfinqa/runner.py:250-260`

The fetch scripts pin upstream revisions
(`scripts/fetch_financebench.py:40-42`, `scripts/fetch_docfinqa.py:15-18`), but the
runners accept `--data-dir`, read gitignored local files, and write no input
manifest. Nothing in the artifact proves the run used the pinned bytes.

**Fix.** Emit an input manifest of paths, sizes, SHA-256 hashes and pinned upstream
revisions, and cover it with the output manifest.

## B5. FinanceBench output manifest omits config, provenance and summary

**Severity: Medium-High.** `csbench/suites/financebench/runner.py:335`

`output_files` receives `evidence_structure.json` and `summary.json` but never
`run_config.json`, `git_provenance.json` or `summary.md`. Same class as finding 4.

## B6. FinanceBench rows are not hash-only

**Severity: Medium-High.** `csbench/suites/financebench/suite.py:289` and `:328`
set `case_id=case.id`, the upstream FinanceBench identifier. The scope guard at
`:368-389` only rejects field *names* containing text-like hints, so `case_id`
passes.

The module describes its rows as hash-only, and the upstream data is CC BY-NC 4.0.
Per-item correctness and evidence geometry can be joined back to the licensed source
by anyone holding the dataset.

**Fix.** Persist a salted identity hash and keep plaintext ids in memory only.

## B7. Truncation arm reports one selected record but claims all of them

**Severity: Medium.** `csbench/arms/truncation.py:80` sets `record_id` to
`request.records[0].id` unconditionally, while `:97` sets
`records_selected=len(request.records)`.

Because FinanceBench record ids are `<case>-p<n>`, this looks like page-granularity
selection, so `evidence_page_selected` (`financebench/suite.py:239`) returns `False`
rather than the `None` its docstring reserves for non-selecting arms. A truncation
tail that genuinely retained the gold page is scored as a miss, which handicaps the
baseline and flatters the selection arms it is compared against.

gpt-5.5 correctly narrowed this: it affects the page-recall diagnostic only, not
answer accuracy or `evidence_in_context`.

**Fix.** Emit `selected=[]` for truncation, or populate it with every record whose
text actually survives.

## B8. Truncation can exceed a budget smaller than its elision marker

**Severity: Medium.** `csbench/arms/truncation.py:46-57`. The `budget_tokens <= 0`
guard at `:49` does not cover the case where the marker alone exceeds the budget:
`head_tail_truncate(text, budget_tokens=1)` returns the marker, whose token estimate
is greater than 1, while the response reports the budget as honored.

## B9. DocFinQA sandbox aborts the run on a missing optional dependency

**Severity: Medium.** `csbench/suites/docfinqa/scorer.py:161` does `import psutil`
inside the try block, but `:192` catches only
`(OSError, subprocess.SubprocessError)`. A missing `psutil` raises
`ModuleNotFoundError`, which escapes and terminates evaluation instead of recording
a runtime error for the item. Raised as speculative by gpt-5.5; confirmed by gemini
and by the orchestrator against the exception clause.

## B10. DocFinQA infeasible rows report `program_correct=False`

**Severity: Low-Medium.** `csbench/suites/docfinqa/suite.py:216`. Unscored
infeasible items are written with `program_correct=False` rather than `None`.
`summarize()` filters on `scored`, so today's aggregate is correct, but any
downstream mean over raw rows counts an unattempted item as a program-synthesis
failure.

---

# Refuted and downgraded

This record is part of the deliverable. Each entry was proposed by one reviewer and
did not survive.

| Claim | Proposed by | Verdict | Why |
| --- | --- | --- | --- |
| TruthfulQA budget silently ignored; `--budgets 500` has no effect | gemini | **Refuted** | The QA runner exposes no `--budgets` or `--max-context-tokens` argument. Verified: no such flag in `qa_runner.py`. The failure path requires a CLI option that does not exist. |
| Closed-book arm corrupts cross-arm token accounting via `tokens_before=0` | gemini | **Refuted** | `financebench/runner.py:274` loops per arm and calls `summarize(rows)` at `:309` on that arm's rows only. No aggregate mixes arms, so nothing is corrupted. `tokens_before=0` is the documented contamination-control definition. |
| TruthfulQA is critical: an arm could delete the context and keep full accuracy | gpt-5.5 | **Downgraded** | The behavior is documented in both the function docstring and `truthfulqa.py:84-88`. Retained as finding 14 at medium, re-scoped to the paper's explanation of *why* inputs were identical. |
| BFCL does not implement the official AST/executable protocol | gpt-5.5 | **Downgraded** | Appendix B discloses this explicitly and calls it the largest deviation, and claims no absolute BFCL score. Disclosed deviation, not a finding. The *matching semantics* underneath it survive as finding 6. |
| SQuAD measures preservation on answerable-only items, not EM/F1 vs gold | gpt-5.5 | **Downgraded** | Appendix B discloses both the unanswerable filter and the preservation framing. Disclosed. The unanchored containment check survives as finding 7. |
| GSM8K accepts the last number anywhere in the completion | gpt-5.5 | **Downgraded** | Appendix B discloses the convention verbatim ("prefer the `####` marker, else the last number, float-tolerant"). Disclosed, and GSM8K is a stated diagnostic control. |
| RULER substring matching can credit a wrong span (`17` inside `117`) | gpt-5.5 | **Downgraded** | Appendix B discloses substring string-match recall, and this is faithful to RULER's own published convention. Deviating would make the harness *less* comparable to upstream. Verified at `ruler/data.py:64-66`. |
| Headline average excludes `niah_single_3` after seeing results | gpt-5.5 | **Downgraded** | Appendix B discloses the exclusion, the reason (truncated reference answers), and reports the 13-task averages including it. Both are published. Disclosed. |
| FinanceBench accepts answers wrong by a factor of 1000 | gpt-5.5 | **Downgraded** | `docs/prereg/financebench_numeric_v1.md` §4.1 defines symmetric display-scale equivalence and §5 states the limitation explicitly and calls it deliberate. Disclosed design choice. Sign errors are *not* rescued, since powers of ten preserve sign. |
| FinanceBench scorer parses `is_percent` but ignores it | gpt-5.5 | **Downgraded** | The prereg §3.4 specifies magnitude-only comparison for percentages. The code implements the prereg. |
| FinanceBench scorer takes the first numeric literal, penalizing chatty models | gemini | **Downgraded** | The prereg §2 freezes "the FIRST numeric literal" and the prompt asks for a bare figure. Disclosed strictness, not permissiveness, and it cannot flatter an arm. |
| DocFinQA/FinanceBench headline accuracy is critical because denominators differ | gemini | **Downgraded** | Real and worth fixing, but the markdown tables print `Scored` and `Infeasible` columns beside the accuracy, and both reports are labelled internal / pipeline-validation-only, not benchmark claims. Retained as a scope B concern at medium, not critical. |
| BFCL numeric collision is critical | gemini | **Downgraded** | Retained as finding 6 at High. Critical requires a demonstrated path to a wrong published conclusion; BFCL's published claim is a null result at p=1.0 that the collision direction does not create. |

Two claims were dropped entirely as unsupported: that the QA runner manifest covers
"only per-item json" (its glob does include per-suite `rows.json` and
`summary.json`; the narrower root-level gap is folded into finding 4), and that
`compare_to_baseline` drops items when an arm is *missing* rows (the published
artifact shows identical `n_paired` across arms in every cell, which is the
duplicate-index signature, not row loss).

---

# Decisions and reversals

Recorded so the reasoning survives the decision.

## D-1. Ruling: correction, not retraction (2026-07-25)

All directions and all significance verdicts hold before and after Holm. Nothing is
withdrawn. The published pooled figure moves and is corrected plainly.

## D-2. Ruling: Option 4, publish on the 2468 survivors; no repair run (2026-07-27)

The 396 destroyed evaluations are unrecoverable. Four options were on the table: a full
re-run of all 2600 items (~$150 at list price), a targeted repair of the 396 ($7.56),
enlarging the drift control first (~$6-9), or reporting on the 2468 survivors ($0).

**Decision: report on the 2468 survivors.** The repair would have fixed a *disclosed and
characterized* limitation by introducing an *undisclosed* one: a mixed-date population
whose re-measured rows carry a known 80-92% item-level agreement rate. Option 3's answer
only matters if the merge happens, and it no longer does. Option 4 is the only path immune
to the unresolved non-determinism question.

## D-3. Reversal: "rather spend than publish a 2468-item figure" (2026-07-27)

Earlier in this audit the maintainer stated a clear preference: "I would rather spend that
than publish a 2468-item figure and revise it later." **That preference is explicitly
reversed.**

The reason is Gate 2. The preference was formed when the only known problem was a smaller
population, and re-measurement looked like a clean way to restore it. Gate 2 then showed
that re-measurement is *itself* noisy: at temperature 0, ten days apart, item-level
continuous-score agreement is 80% (full_context), 88% (needlepath) and 92% (compresr).
The full_context arm performs no selection, so that movement is the answering model, not
the harness. Spending to re-measure would therefore not have bought a cleaner number; it
would have traded a quantified, disclosable population limitation for an unquantified
temporal one.

The reversal is recorded rather than quietly applied because the original preference was
stated on the record and was reasonable on the information available at the time. What
changed was the evidence, not the standard.

---

# Instrument failures

A recurring failure mode surfaced during this audit, distinct from the defects
themselves: a confident number produced by an instrument that was never validated,
caught only by contradiction with something independently known. It occurred nine
times. Recorded because the pattern is more useful than any individual instance.

The durable fixes were never vigilance. In every case the thing that closed the
hole was making the wrong call impossible to express: a required `expected_n`
argument, a declared grid, a frozen-figures manifest, a byte-identity gate.
Vigilance failed within minutes of writing a docstring warning against the exact
error it warned about.

| # | Instrument | Wrong output | Caught by |
| --- | --- | --- | --- |
| 1 | "byte-identical duplicates therefore lossless" | missed that 396 evaluations were destroyed | maintainer checking distinct positions |
| 2 | hand-rolled token reconstruction | 0/2468 match, read as dataset divergence | offsets were a constant ~22 tokens, the size of a record wrapper |
| 3 | guard-reachability counter | 4 unreachable guards, two of them wired | contradicted code written minutes earlier |
| 4 | `reduction_ratio` as the fallback token metric | "fallback saves more than engaged", incoherent | the claim is impossible on its face |
| 5 | single-line grep for "orders of magnitude" | reported absent; it was line-wrapped | re-check with different pattern |
| 6 | inferred bound in `forbid_index_keying` | crashed on every damaged deposit | the byte-identity gate |
| 7 | cross-file consistency regex | "N=2468 absent from paper"; it appears 4x as `$2{,}468$` | manual check before acting |
| 8 | run032 Headroom artifacts | fast-path bounds that *agreed* with the claim, from the wrong run | run means did not match the paper's |
| 9 | `gh pr create --dry-run` | "Would have created a Pull Request" for a PR the API rejects with 422 | probing the REST endpoint directly |

Entry 9 is the only one where the unvalidated instrument was a vendor tool rather
than something written here. `gh pr create --dry-run` performs no server-side
validation, so it reports success for a head/base combination GitHub refuses. A
dry run that does not exercise the failing layer is not a rehearsal.

Entry 8 is the most instructive: the wrong-run numbers were close enough to the
claim they were meant to check that they read as confirmation. An instrument that
agrees with you is not thereby validated.

**Standing rule adopted from this audit:** validate an instrument against a
known-positive and a known-negative before reporting its output. Every guard and
detector added in v1.1 ships with both controls, and the reachability detector
additionally documents what AST matching cannot see, so a green run is not read
as proof of runtime coverage.

## Single-producer guards

A second failure class, distinct from the instrument failures above and from the
inferred-bound error at entry 6. It concerns guards rather than measurements, and it
recurred within this harness, including inside the guards written to close the
row-duplication defect itself.

**The pattern: a guard that derives its condition from one producer of that
condition will miss every other producer.** The guard is not wrong about the case it
was written from. It is silently blind to the others, and it reports success while
blind, because the condition it evaluates is a proxy for the real one — a single
producer standing in for the set.

| # | Guard | Condition derived from | Producers missed |
| --- | --- | --- | --- |
| A | index-keying tripwire (`csbench/suites/ruler/pairing.py:47`) | the one call site it was wired into, the resume cache (`harness.py:472`) | every other path that keys rows for a join or a cache; the comparison join that carried the original finding is not among its callers |
| B | write-time item identity (`harness.py:476-483`) | `len(rows)`, i.e. the row set under test | any truncation that removes whole rows — 99 rows is 99 positions, so a truncated prefix validates as complete. Closed by requiring `expected_n` from the caller |

Entry B is the instructive one: the guard was added *by the fix* for the defect it
checks, and in its first form would have certified the damaged deposit as sound. A
guard written while thinking about one producer encodes that producer, not the
invariant.

**Distinguished from entry 6.** The inferred-bound error is a guard computing a
*threshold* from data it should have been given. This class is a guard computing its
*trigger* from one member of a set. The remedies differ: entry 6 is fixed by
requiring the bound as an argument; this class is fixed by deriving the condition
from the invariant itself — assert on the property, not on the one symptom that
happens to produce it.

**Standing rule adopted from this audit:** when a guard tests a condition that more
than one code path can produce, enumerate the producers before writing the test, and
either cover the set or state in the guard which producers it does not cover. A guard
that names one producer in its own implementation is scoped to that producer,
whatever its docstring claims.

---

# Disproven claims

Claims made during this audit that were subsequently disproven. Recorded with the same
weight as the findings.

### D1. "RULER paired statistics are computed on a corrupted population"

**Status: disproven.** Proposed independently by both reviewers, confirmed by both under
cross-refutation, and rated Critical by the orchestrator on the strength of the
published `n_paired < 100` values.

Disproven by recomputing all 52 comparisons from the deposited per-item rows: the
published index join and correct position pairing agree in **52 of 52 cases**, with zero
significance flips, because the duplicate rows are byte-identical. `n_paired` was the
correct distinct-item count all along.

**Why three reviewers got it wrong.** All of us reasoned from `ruler/stats.py:201`'s
warning that `index` is non-unique to the conclusion that an index-keyed dict must lose
information. That inference is only valid if the collided rows differ. Nobody checked
whether they differed until the recomputation. The symptom (`n_paired < 100`) was
consistent with both the wrong hypothesis and the right one, and it was never
discriminating evidence.

### D2. "The headline replication statistics were produced by an uncommitted script, and
this is a JOSS reproducibility defect"

**Status: disproven and withdrawn.** Asserted during the recomputation on the basis that
no pooled computation exists on any branch of the public repository and no committed
public artifact contains `+4.99`, the CI bounds, or the per-length p-values.

Disproven by evidence:

- The flagship analysis is committed internal code at
  `src/core/agentbuilder/selective_state/needlepath_bench/ruler_stats.py` in the
  `nextmoca` repository, present on five branches (`needlepath/ruler-benchmark`,
  `needlepath/tscg-adoption`, `needlepath/headroom-arm`,
  `needlepath/headroom-032-rerun`, `feat/excerpt-budget-redistribution`), with a
  companion `tests/needlepath_bench/test_ruler_stats.py`.
- Its outputs are committed under `docs/needlepath/data/`, including
  `official_ruler_full_run_v1_item_identity_report.json`.
- That module is the 801-line original of which the public
  `csbench/suites/ruler/stats.py` is the extracted copy.
- The public repository ships `csbench/stats.py` and `csbench/suites/ruler/stats.py`, so
  the statistical primitives are public.

**Why the claim was wrong.** The search was scoped to the public repository's branches.
Finding nothing there licensed no conclusion about whether the method was committed
elsewhere, and one was drawn anyway. Absence of evidence within an arbitrary boundary
was treated as evidence of absence. The "JOSS reproducibility defect" framing that
followed from it is withdrawn in full and appears nowhere else in this document.

The legitimate residue is narrower and is not a defect claim: there is no *public* path
from the deposited per-item rows to the replication table. That is addressed as a
proposal in the sequencing section, not as a finding.

---

# Recommended sequencing

1. **Decide on findings 1b and 2** before anything else. They are the only findings that
   change what a published number means, and they are one supersession decision, not a
   patch. No claim loses significance; direction holds throughout. Findings 3 and 4
   attach to the same decision.
2. Findings 6 and 7 are scorer-level and affect published control results. They need a
   re-score to size, which is also a supersession question. Finding 5 is latent and
   does not.
3. Findings 8, 9, 10, 12, 13 are code fixes with no published-number impact and can
   proceed normally.
4. **Proposed, not a finding: a public aggregation tool.** Add
   `csbench/suites/ruler/aggregate.py` so an outside reader can regenerate the
   replication table (per-length and pooled) from the deposited per-item rows using only
   public-repo code. Read-only over `items/**/*.jsonl`, importing only `csbench.stats`
   for `mcnemar_test` and `bootstrap_ci_delta`. It should dedup on `(position, index)`
   requiring byte-identical copies, assert identical `[(position, index)]` sequences
   across arms, report deduplicated figures as primary with the raw duplicate-retaining
   figures as a labelled secondary and `n_distinct` beside `n_rows`, take the task set
   as an explicit argument so neither the 13-task nor 12-task variant is a hidden
   default, and apply Holm across the co-primary family emitting raw p, adjusted p and
   the threshold each was compared against. Writes to a user-supplied `--out`, never
   into the deposit. Full specification in
   `scratchpad/RECOMPUTATION_ruler_repl_v1.md` section 6.
4. Scope B findings should be resolved before those branches merge, with B3 (the
   prereg mismatch) settled before any scoring run on that protocol.

Nothing in this document has been acted on.
