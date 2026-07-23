# context-selection-bench

A reproducible, matched-protocol benchmark harness for **context-selection and
prompt-compression methods** on public evaluation suites.

## What this is

The harness runs any context-reduction method as a uniform **arm** (the same
records and task go in, selected/compressed context and timing metadata come
out), so different methods are compared under one protocol, one set of models,
one set of scorers, and one set of seeds. It ships:

- the harness and, for each suite, a reimplementation of that benchmark's
  published reference metric (never a bespoke metric of ours; per-suite deviations
  are documented in the paper's scorer-provenance appendix);
- a paired-statistics module (McNemar + bootstrap confidence intervals);
- competitor arms that run locally (full-context passthrough, and compression
  baselines);
- a **hosted-adapter client** for methods served over HTTP, with an interface
  specification and a contract-conformance stub server;
- pinned configs/seeds, committed **sha256 manifests**, and a tool to verify
  published per-item results against them;
- fetch-scripts (with license notes) for each upstream dataset: upstream data
  is **not** redistributed here.

## Suites (v1)

| Suite | Module | Runner |
|---|---|---|
| RULER (long context) | `csbench.suites.ruler` | `python -m csbench.suites.ruler.harness` |
| SQuAD v2 · BFCL · TruthfulQA | `csbench.suites.{squad,bfcl,truthfulqa}` | `python -m csbench.suites.qa_runner` |
| GSM8K · GSM-IC (diagnostic) | `csbench.suites.gsm` | `python -m csbench.suites.gsm.runner` |

GSM8K/GSM-IC is a **diagnostic boundary suite**, not a flagship result: see
`csbench/suites/gsm/README.md` for why.

## The arm contract

Every arm implements one interface: `select(request) -> response`. Local arms
run in-process; hosted methods are reached through an HTTP client against an
endpoint the method's provider operates. See `INTERFACE.md` for the
request/response schema and reproducibility notes.

## Reproducing results

Every competitor arm and the full-context arm reproduce end-to-end with your own
API keys; the hosted `needlepath` arm additionally requires the endpoint URL.

```bash
# 1. install (add extras for the arms you run)
pip install -e ".[dev,gemini]"          # + ,llmlingua2  or  ,cpc  for those arms

# 2. fetch a suite's dataset (fetch-scripts only, no data is redistributed here)
#    see DATASETS.md ; RULER lands under data/ruler/<length>/<task>/test.jsonl

# 3. provide your model key
export GEMINI_API_KEY=...

# 4. a running-spend hard stop is mandatory for any real run
export BENCH_BUDGET_USD=50

# 5. run the RULER surface (full-context control + hosted needlepath)
python -m csbench.suites.ruler.harness \
  --arms full_context,needlepath \
  --needlepath-url https://<hosted-endpoint> --operating-point np-2026-07-r1 \
  --n 100 --lengths 8k,16k --out runs/ruler
#   add llmlingua2 / cpc to --arms to run the local compression baselines

# 6. verify published per-item outputs against a committed manifest
python tools/verify_manifests.py <manifest.sha256> --base-dir <downloaded-outputs>
```

The runner estimates spend up front and **aborts before any API call** if it
would exceed `BENCH_BUDGET_USD`. Per-item outputs, an aggregate matrix, and a
sha256 manifest are written under `--out`.

## Licensing

- **Code:** Apache-2.0, see [`LICENSE`](./LICENSE).
- **Result data:** CC BY 4.0, see [`LICENSE-DATA`](./LICENSE-DATA).
- **Upstream datasets:** obtained via fetch-scripts; each retains its own
  upstream license and is not redistributed here.

## Citation

If you use this benchmark program, please cite the archived artifacts as the
canonical reference:

> *A Matched-Protocol Benchmark Program for Context Selection in Agentic LLM
> Systems: harness and artifacts.* Next Moca Global, Inc., 2026. Zenodo.
> DOI: [10.5281/zenodo.21502503](https://doi.org/10.5281/zenodo.21502503).

See [`CITATION.cff`](./CITATION.cff) for the author list and machine-readable
citation metadata. An arXiv preprint of the accompanying paper is forthcoming.
