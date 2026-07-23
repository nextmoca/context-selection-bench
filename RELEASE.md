# Release v1.0.0

This repository accompanies the paper **A Matched-Protocol Benchmark Program for Context Selection
in Agentic LLM Systems** (Next Moca Global, Inc.). See `CITATION.cff` for the author list and
citation.

> This file describes the v1.0.0 release as a whole: the GitHub repository plus the archived Zenodo
> bundle (DOI 10.5281/zenodo.21502503). Per-item run data (the `runs/` and deposit trees named below)
> is not stored in the git repository; it lives in the Zenodo archive and is regenerable with the
> harness. Cloning the repository gives you the harness, configs, scorers, and manifests, not the
> per-item run data.

- Paper: arXiv preprint forthcoming (cs.CL).
- Archived artifacts: DOI [10.5281/zenodo.21502503](https://doi.org/10.5281/zenodo.21502503).
- Code: Apache-2.0 (`LICENSE`). Result data: CC BY 4.0 (`LICENSE-DATA`). Upstream datasets are
  fetched by scripts, never redistributed here, and remain under their own licenses (`DATASETS.md`).

## Contents

- `csbench/`, the extracted public harness: arm contract, suites, reference-metric scorers (each
  benchmark's published metric, reimplemented here; deviations enumerated in the paper's
  scorer-provenance appendix), and the four latency measurement classes.
- `runs/ruler_repl_v1/`, the public-harness replication run: per-item outputs and a SHA-256 manifest.
- `tools/verify_manifests.py`: recompute every file's SHA-256 against its manifest.
- `scripts/`: dataset fetch scripts (RULER generator, etc.).
- `paper/`: LaTeX source of the paper.

## Reproduce the replication result

```bash
pip install -e ".[dev]"
python scripts/fetch_ruler.py                 # see DATASETS.md
# reach the context-selection system via the hosted adapter (operating point np-2026-07-r1); see INTERFACE.md
python tools/verify_manifests.py runs/ruler_repl_v1/manifest.sha256 --base-dir runs/ruler_repl_v1
```

## Scope of released data

This is a mixed-license deposit assembled by `tools/build_deposit.py`; `DEPOSIT.md` gives the full
recipe and `NOTICE` the third-party attributions. Every run ships a SHA-256 manifest. Per run:

| Run(s) | Per-item prompts | What ships | License reason |
|---|---|---|---|
| RULER (`ruler_v1`, `ruler_repl_v1`, `gate_ruler_v1`) | hash-only | completions, scores, token counts, metadata; qa answers hashed | haystacks embed all-rights-reserved essays + CC BY-SA SQuAD/HotpotQA; regenerate prompts via `fetch_ruler.py` |
| AppWorld (`appworld_v1`, `gate_appworld_v1`, `gate_probe_v1`) | none | aggregates + paired-stats + content hashes | AppWorld no-plaintext terms + dataset canary |
| `controls_v1`: TruthfulQA, BFCL | plaintext | upstream items + our outputs | Apache-2.0 permits redistribution with attribution (`NOTICE`, `LICENSES/`) |
| `gsm8k_v1` (separate full-set run) | hash-only | full-set aggregate + manifest | figure is from a separate full-set run, not the `controls_v1` harness; GSM8K is MIT, regenerate per-item via the harness |
| `controls_v1`: SQuAD v2 | hash-only | outputs + scores | CC BY-SA share-alike would conflict with this deposit's CC BY 4.0 |

Gate engine-internal signals, the AppWorld canary, and all prompt text are excluded by construction
(see `DEPOSIT.md`).

## Cite

See `CITATION.cff`.
