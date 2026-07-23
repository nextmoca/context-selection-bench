# Contributing

Thanks for your interest in improving the benchmark. Contributions are welcome
via fork and pull request. Direct pushes to `main` are not accepted.

## How to contribute

1. Fork the repository and create a branch in your fork.
2. Make your change. Keep pull requests focused: one topic per PR.
3. Run the checks locally before opening the PR:

   ```bash
   pip install -e ".[dev]"
   pytest -q                          # tiny-N smoke tests, no GPU, no keys
   python scripts/scrub_check.py --root .
   ```

4. Open a pull request against `main`. CI must pass, and a maintainer
   (see `.github/CODEOWNERS`) must approve before merge. Maintainers are the
   only people who can merge.

## What we are looking for

- New suites or arms that follow the existing arm contract (see
  `INTERFACE.md`): same items, same scorers, same protocol for every arm.
- Fixes to scoring, harness correctness, or reproducibility.
- Documentation improvements.

## Ground rules for benchmark integrity

- Results claims must be reproducible: a PR that adds or changes reported
  numbers must include per-item outputs and a SHA-256 manifest, generated the
  same way as the released runs (`tools/verify_manifests.py` must pass).
- No cherry-picking: report the pre-specified metrics for the whole suite, not
  a favorable subset. Negative and null results are welcome.
- Upstream datasets are never committed to this repository. Add a fetch script
  with license notes instead (see `DATASETS.md`).
- Do not include secrets, API keys, or internal identifiers in code, comments,
  or commit messages. CI runs a scrub gate over the tree and commit history.

## Reporting a result you cannot reproduce

Open an issue using the reproduction report template. Include the run id, your
environment, and the exact command. We treat reproduction failures as bugs.

## License

By contributing, you agree that code contributions are licensed under
Apache-2.0 and result data under CC BY 4.0, matching the repository's existing
license split.
