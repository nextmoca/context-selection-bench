# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
