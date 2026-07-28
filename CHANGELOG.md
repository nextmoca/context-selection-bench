# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## 1.1.0 - 2026-07-27

Corrective release. A row-duplication defect in the RULER replication harness is
fixed, the replication population and figures are restated, and the public
dataset-regeneration script is repaired. Direction and significance of every
reported finding are unchanged.

### Fixed
- RULER harness resumed by RULER's `index` field, which is not unique within a
  task. On resume, every position sharing an index received the same row object,
  which was mutated in place and written once per position, so byte-identical
  copies replaced the evaluations they displaced. 396 of 7,800 rows in 42 of 78
  cells of `ruler_repl_v1` were lost this way and are not recoverable. Pairing
  and resume now key on `position` through a single shared utility
  (`csbench.suites.ruler.pairing`), which asserts item identity and refuses to
  intersect a mismatched population. Reading a pre-fix damaged cell file now
  fails loudly, which is intended. A regression test asserts one row per
  position 1..n and fails against the damaged artifacts.
- `compare_to_baseline` joined arms on the same non-unique field and took a set
  intersection, so a shrinking paired count was reported as a smaller population
  rather than raised as a defect.
- `scripts/fetch_ruler.py` could not generate the dataset and reported success
  anyway. The tokenizer shim it patches into RULER defined `text2tokens` /
  `tokens2text`, while RULER's generators call `text_to_tokens` /
  `tokens_to_text`, so every task failed; `prepare.py` swallowed the error and
  the script then printed a success line and exited zero having written nothing.
  A relative `--out` also resolved inside the RULER clone rather than the
  caller's directory. The shim now uses the upstream API and asserts it against
  the pinned commit, every generated cell is verified to exist with the expected
  row count, paths are resolved, and tests cover the original silent failure.
- Output manifests covered only the per-item rows, leaving `matrix.json`,
  `matrix.md`, the combined summaries, `run_config.json` and
  `trusted_benchmark_manifest.json` unhashed. The published
  `runs/ruler_repl_v1/manifest.sha256` covers 78 of 85 files, and the headline
  figures live in one it omits. All generated artifacts are now covered.
- Replication cost figures were computed on inconsistent populations: the
  Needlepath cost matched a deduplicated token total while the full-context cost
  matched a raw one. Both are now computed on one population: Needlepath
  $15.67, full context $23.29, Compresr $20.70 ($18.03 API plus a $2.68
  compression fee). Mean input tokens 11,788 to 7,929, a 32.7% reduction for
  Needlepath and 22.6% for Compresr.
- `paper/main.tex` and the statistical-methods appendix described the pooled
  accuracy delta and the binarized-correctness bootstrap the wrong way round.
  The pooled accuracy delta is the continuous mean fractional score; the
  companion figure is on binarized correctness. Corrected in both places,
  including a stale Compresr sentence that carried the old value under the old
  label.
- The RULER latency table labelled its cells "per-item preparation latency"
  while every cell is a mean over a right-skewed distribution. Relabelled as
  mean per-item, with a note that means exceed medians throughout and that no
  cell should be read as a typical item's cost. Headroom's cell blends a fast
  lossless-fold path and a slower ML-inference path; the caption now states the
  split rather than disclosing only the slow end, and "worst item 7.3 s" is
  corrected to "worst task-cell mean", which is what the source table records.
  The ~400x figure versus CPC is retained and labelled a ratio of means.
- The 40% fallback rate was reported without its mechanism, which invited
  attribution to the operating point's envelope gate. The gate runs in shadow
  at np-2026-07-r1 and emits telemetry only; fallback is an evidence-coverage
  obligation, and all 994 fallback rows carry selection_safe=false with an
  evidence or answerability reason. The claim that fallback items "save zero
  tokens and are scored identically to full context" is half right and is now
  split: billed input tokens are identical on all 994, verified item by item,
  but scores agree on only 87.5% (continuous) and 93.7% (binarized), because
  separate model calls on identical prompts diverge.
- `scripts/fetch_ruler.py` skipped re-patching a RULER clone that carried the
  pre-1.1 patch marker, which the broken shim also used, so an existing clone
  stayed broken. The marker check now verifies the tokenizer API and fails with
  instructions rather than silently skipping.

### Changed
- Replication figures restated on the 2,468 items actually evaluated, replacing
  figures computed on the 2,600 rows as stored. Metric and task scope are
  unchanged (13 tasks, continuous mean fractional score for the delta and CI,
  binarized correctness for McNemar). Needlepath versus full context: pooled
  accuracy delta +4.62pp [+3.32, +5.94], pooled McNemar p = 0.0013; per length
  +4.48pp (raw p = 0.031) at 8K and +4.77pp (raw p = 0.022) at 16K,
  Holm-adjusted p = 0.044 at both lengths, significant at both lengths under
  Holm. On binarized correctness, +2.11pp [+0.89, +3.36]. Compresr versus full
  context: pooled +1.58pp [+0.11, +3.05], p = 0.166, not significant at either
  length. Average input tokens 11,788 to 7,929 (-33%).
- The loss is symmetric across arms and unbiased within task, but it is
  task-reweighted: collisions concentrate in the hard `niah` tasks, which lose
  proportionally more items and are where the selection arm's advantage is
  largest. That reweighting is the most likely reason the pooled delta moved
  down. Stated in a new disclosure section in the paper.
- The correction comes from the population, not from a different computation:
  run over the rows as stored, the aggregation tool reproduces the previously
  published point estimates and McNemar p-values exactly. It does not reproduce
  the published bootstrap interval bounds exactly, and cannot: an interval
  depends on item iteration order as well as on the data and the seed, and the
  deposit does not record the order the original analysis used. The bounds it
  emits differ in roughly the third decimal, which is the same effect described
  under iteration order below.

- The public aggregation tool reported the arm's internal
  `tokens_before`/`tokens_after` ratio as "token reduction" (Needlepath 67.98%)
  while the paper reports the billed basis against the matched full-context arm
  (32.7%), and it applied a single blended token rate, so it could not
  reproduce Compresr's cost, which includes a non-token compression fee. It now
  reports the billed reduction as the headline figure with the internal
  selection ratio beside it, and supports an arm-specific service fee. Run over
  the deposit it reproduces the paper's economics: Needlepath 32.74% / $15.67,
  Compresr 22.61% / $20.71, full context $23.29. The published Compresr cost of
  $20.70 is the rounded sum recorded upstream ($18.03 API + $2.68 fee); the
  unrounded recomputation is one cent higher.
- The completeness check inferred the expected arms from the deposit it was
  validating, so deleting an entire arm passed and silently removed a published
  comparison. Arms are now declared alongside tasks and lengths.
- The write-time item-identity guard took its expected count from the rows it
  was checking, so a truncated prefix of positions 1..99 was "complete". It now
  takes the planned example count from the caller.
- The pooled McNemar was described as the confirmatory aggregate test while the
  declared primary test is the pair of co-primary per-length Holm-corrected
  tests. On the corrected figures the pooled result (p = 0.0013) is numerically
  stronger than the Holm-adjusted per-length result (p = 0.044), so that label
  would have promoted the more favourable statistic after seeing both. Pooled
  McNemar is secondary and descriptive throughout, and the paper now says why
  the distinction is being flagged.
- The limitations section still carried the retracted claim that fallback items
  "score as full context". Corrected to the verified claim: identical billed
  input tokens on all 994, with 87.5% continuous and 93.7% binarized score
  agreement.

### Added
- `runs/drift_control_2026-07-27/`: the 150 paired before/after evaluations, the
  per-arm summary, and the selection and analysis scripts behind the decision
  not to re-run the destroyed items. The decision rested on numbers with no
  released artifacts; they are now auditable, including the seed and the 50
  selected item identities.
- Per-row `input_sha256` on every RULER item. The deposited artifacts recorded
  derived quantities but no input text and no input hash, so when the dataset
  was regenerated ten days later, byte-identity could only be inferred from
  agreeing fingerprints rather than checked. Recording the hash makes a future
  regeneration checkable byte-for-byte. The digest over the 2,600 regenerated
  inputs used for this release is
  `5273de7aeb38dc0f41b9c6d17fea7d1d...` (SHA-256, truncated).
- `tests/test_guard_reachability.py`: a structural assertion that every guard in
  the package is called from a production path. An inventory found the harness's
  pre-existing guards all reachable, and the two unreachable ones both
  introduced by this release's own fix for the defect they were meant to
  prevent; they existed only in their unit tests. Both are now wired. The
  detector is validated against a known-wired and a known-unwired guard, since a
  detector that cannot fail is not evidence.
- `tests/frozen_figures.py` and `tests/test_frozen_figures.py`: headline figures
  published on live claim surfaces, asserted against the paper. The manifest
  states in its own header that it is a derived copy of the tracker registry,
  that the website keeps a second copy neither repository can import, and that
  reconciliation between them is a human step no test performs. Interval bounds
  round outward (floor the lower, ceil the upper) so a published interval always
  contains the computed one; the rule is machine-checked by recomputing it from
  the unrounded bounds rather than stated in prose.
- `csbench.suites.ruler.aggregate`: regenerates the replication table from
  deposited per-item rows using only public code, so the deposit can be
  recomputed and not merely verified against a manifest. Reports accuracy and
  EM deltas with CIs, raw and Holm-adjusted McNemar per length and pooled,
  token reduction, mean input tokens, fallback rate and cost. Task scope is an
  explicit argument. The damaged as-stored population is available only behind
  `--legacy-order-pairing`, off by default, with a stderr banner and a
  provenance field marking its output as not a corrected result.
- `csbench.stats.holm_adjust`: the Holm correction named as the primary decision
  rule in the statistical methods but not previously implemented anywhere in the
  package. Verdicts on the published and corrected figures are unchanged.
- Dataset regeneration result reported in the paper: ten days after the original
  run, on different hardware, the pinned generator (commit 38da79d7, seed 42)
  matched all 26 task/length cells and all 7,404 surviving rows on four
  independent fingerprints: RULER's index, the expected answer, the generated
  haystack length, and the tokenized prompt length recomputed through the
  harness's own request-construction path. A 100.00% match on all four. Not a
  byte comparison: the deposit stores no input text and no input hash, so
  byte-identity cannot be established retrospectively.
- Drift control reported in the paper as a descriptive methodological finding:
  re-running 50 surviving items across three arms ten days later gave item-level
  exact agreement of 80%, 88% and 92% on the continuous score and 88%, 94% and
  94% on binarized correctness, with no paired test significant and selection
  behaviour unchanged. Reported descriptively, with the caveat that n = 50 per
  arm cannot separate non-determinism at temperature 0 from model-side change
  and that the threshold for calling it material was chosen rather than derived.

- Bootstrap interval bounds depended on item iteration order, which nothing
  pinned; the intervals were deterministic by accident. The aggregation tool now
  pins iteration order explicitly (tasks sorted, lengths ascending) and a test
  asserts the declared grid cannot leak into the computation order. Found by the
  byte-identity gate on the corrected table, which failed after an unrelated
  change to how the expected task set was declared. Point estimates and McNemar
  p-values are order-independent and were unaffected. The paper's methods now
  state that a replicator iterating in a different order should expect interval
  bounds to move in roughly the third decimal with no verdict changing.

### Verified unchanged
- The deposited replication run was complete. The corrected aggregation now
  refuses to compute on a missing cell rather than skipping it, and re-running
  it produced figures byte-identical to those computed before that check
  existed. Every selected length/task/arm cell was present; the completeness
  guard never fired on real data.
- `ruler_v1` and every figure derived from it, including the five-arm ordering.
  That deposit stores all arms per item file, so no join by `index` occurs; it
  was verified clean at 2,600 files with zero duplicate or missing positions.
- AppWorld, the clean-desk controls, GSM8K, and all gate runs.
- The direction and significance of every reported finding.

## 1.0.1 - 2026-07-23

### Changed
- Clarify scorer provenance: reference-metric reimplementations, deviations
  enumerated; results unchanged. The five bundled suites (RULER, SQuAD v2, BFCL,
  TruthfulQA, GSM8K) are scored by reimplementations of each benchmark's published
  reference metric, never a bespoke metric of ours, not by the upstream official
  scoring packages; only the datasets are obtained from official sources. Every
  per-suite deviation is now enumerated in a scorer-provenance appendix in the
  paper. AppWorld continues to be scored by its genuine official evaluator. The
  wording in `paper/main.tex`, `README.md`, `RELEASE.md`, and the suite module
  docstrings now states this precisely. No code paths, runs, or reported numbers
  changed.
- Citation metadata now uses the Zenodo concept DOI (`10.5281/zenodo.21502502`,
  all-versions) in `README.md` and `CITATION.cff`, so citations always resolve to
  the latest release. The paper's own artifact references (the title-page data
  availability note and the per-claim provenance table) stay pinned to the exact
  version deposit.

## 1.0.0 - 2026-07-23

- Initial public release.
