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
| AgentDojo (prompt injection, agentic) | `csbench.suites.agentdojo` | `python -m csbench.suites.agentdojo run` |
| AgentDyn (prompt injection, agentic; published run) | `csbench.suites.agentdyn` | `csbench/suites/agentdyn/reproduce.sh` |
| AutoDojo (adaptive prompt injection; published run) | `csbench.suites.autodojo` | `csbench/suites/autodojo/reproduce.sh` |

GSM8K/GSM-IC is a **diagnostic boundary suite**, not a flagship result: see
`csbench/suites/gsm/README.md` for why.

The AgentDojo suite is a paired harness (baseline vs. selection at the
tool-result hop) with whole-arm validity gates and a summarizer that refuses
incomplete runs; see `csbench/suites/agentdojo/README.md`.

## Agent security suites

Three prompt-injection benchmarks, each run from a fork pinned at an exact
commit, in two arms: the benchmark's own agent (`none`) and the same agent
with the `next_moca` defense element after the tool executor. The element is a
transport: it sends each batch of tool results to the Next Moca hosted service
and passes on what the service returns, and the agent sees the original tool
output on any error. Each suite directory carries the pinned commit, the run
configuration, a reproduction script, the scorer and the results.

| suite | attacked tasks | attack success, without → with Next Moca | tasks completed under attack, without → with Next Moca |
|---|---|---|---|
| AgentDojo | 949 | 268 (28.2%) → 0 (0.0%) | 56.2% → 71.3% |
| AgentDyn | 560 | 217 (38.8%) → 35 (6.3%) | 56.4% → 57.5% |
| AutoDojo (adaptive) | 144 | 93 (64.6%) → 17 (11.8%) | 81.9% → 80.6% |

- AgentDojo: [`csbench/suites/agentdojo/published`](csbench/suites/agentdojo/published/run_config.json).
  `gpt-4o-2024-08-06`; AgentDojo v1.2.2, suites workspace, travel, banking,
  slack; attack `important_instructions`; n = 949 attacked tasks per arm; Next
  Moca hosted service, October 2026, build `6c0eaba7bc4aa35c`; run 2026-10-26.
- AgentDyn: [`csbench/suites/agentdyn`](csbench/suites/agentdyn/README.md).
  `gpt-4o-2024-08-06`; AgentDyn suites shopping, github, dailylife
  (benchmark version v1.2.2); attack `important_instructions`; n = 560
  attacked tasks per arm; Next Moca hosted service, November 2026, build
  `42aea840d7689d36`; run 2026-11-10.
- AutoDojo: [`csbench/suites/autodojo`](csbench/suites/autodojo/README.md).
  `gpt-4o-2024-08-06`; AgentDojo v1.2.2 banking suite; injections optimized
  separately against each arm (`gemini-2.5-pro` optimizer, 4 iterations, 5
  variants, variant 0 benchmarked); n = 144 attacked tasks per arm; Next Moca
  hosted service, November 2026, build `bbf91c3e845c6cc8`; run 2026-11-10.

Attack success is the benchmark's own `security` check (the injected goal was
achieved). Tasks completed under attack is the benchmark's `utility` check on
the same attacked tasks. Each arm ran once; the counts are single samples.
`python -m csbench.suites.agentdyn check` (likewise `autodojo` and
`agentdojo.published`) re-validates the committed numbers offline.

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
> DOI: [10.5281/zenodo.21502502](https://doi.org/10.5281/zenodo.21502502).

See [`CITATION.cff`](./CITATION.cff) for the author list and machine-readable
citation metadata. An arXiv preprint of the accompanying paper is forthcoming.
