# AutoDojo suite (published GPT-4o run)

[AutoDojo](https://github.com/xhOwenMa/AutoDojo) is a generative benchmark
built on AgentDojo and AgentDyn. Instead of a fixed attack, it optimizes
prompt injections against a given agent and defense, observing only whether
each candidate succeeds, and then benchmarks the optimized injections. This
directory pins the code that produced the published GPT-4o result on the
banking suite, carries that result, and shows how to run it again.

## Pinned code

| | |
|---|---|
| upstream | `xhOwenMa/AutoDojo` at `aa458794b7eb7f082e94daeac4c0f55bfc0c90cc` (MIT) |
| fork | [`nextmoca/autodojo`](https://github.com/nextmoca/autodojo), branch `next-moca` |
| pinned commit | `8bf235016dbd4dfd92e0a6476d87791e0919bb60` |
| defense element | `next_moca/np_next_moca.py`, sha256 prefix `b9698486c20eb54d` |
| plugin | `next_moca/np_autodojo_plugin.py`, sha256 prefix `c84ebd37608d1814` |

The run used a clean checkout of upstream `aa458794` with the two files above
on `PYTHONPATH` (the run log records both hashes). The pinned commit adds those
files byte for byte and changes no upstream file except `README.md`. The
plugin registers the element through AutoDojo's `AGENTDOJO_DEFENSE_PLUGINS`
hook between the tool executor and the model, and sends `openai/<model>`
targets to the OpenAI API directly. The element sends the trailing batch of
tool results to the hosted service and passes on the context the service
returns; on a timeout, an HTTP error, a malformed response or a server
pass-through directive the model sees the original tool output.

## Run configuration

Full detail in [`run_config.json`](run_config.json).

- Target model: `gpt-4o-2024-08-06` (OpenAI) in both arms.
- Suite: banking (16 user tasks x 9 injection tasks), AgentDojo `v1.2.2`;
  attack `autodojo`.
- Optimizer: `gemini-2.5-pro` as analyzer and rewriter, black box
  (success-only feedback), 4 iterations, 5 variants, variant 0 benchmarked.
  Injections are optimized separately against each arm's own pipeline, so the
  `next_moca` arm faces injections optimized against it. The AutoDojo paper
  uses 8 iterations and a different optimizer model.
- Population per arm: 144 attacked tasks and 16 benign tasks.
- Arms: `none` (no defense) and `next_moca`.
- Service: Next Moca hosted service, November 2026, build `bbf91c3e845c6cc8`
  (the same build id at the start and the end of the run), operating point
  `np-2026-10-r8`.
- Dates: 2026-11-10 (UTC).

## Results

[`results/summary.json`](results/summary.json). Counts are tasks.

| suite | attacked tasks | attack success, none → next_moca | tasks completed under attack, none → next_moca | benign tasks completed, none → next_moca |
|---|---|---|---|---|
| banking | 144 | 93 → 17 | 118 (81.9%) → 116 (80.6%) | 14/16 → 14/16 |

Per-row result files and the optimized injection caches for this run are not
included in this release; the summary records the counts the scorer produced
from the rows.

## Checking and re-scoring

```bash
pip install -e ".[dev]"
python -m csbench.suites.autodojo check     # offline consistency check of run_config.json and results/summary.json
python -m csbench.suites.autodojo score --none <logdir> --next-moca <logdir> --build-id <id> --out <dir>/summary.json
```

For this suite `check` can confirm only that the committed summary matches the
pinned configuration, the planned population and its own arithmetic: without
the per-row files it cannot re-derive the counts. `score` reads
AgentDojo-format result rows under each arm's log directory and refuses to
produce a summary when a row is incomplete, unreadable, not boolean-scored or
marked with a provider error; when a case appears twice; when a suite is not
the full planned user-task by injection-task population; when the two arms did
not score the same cases; or when the directory given for an arm does not hold
that arm's rows (each arm's pipeline must be the configured one; the defended
arm must carry well-formed element telemetry and the other must not; a
defended arm that never reached the service is refused).
Task ids must be exactly the planned ids listed in `run_config.json`.
`--build-id` is the build id the service reported for the run, and
`--operating-point` (required) must match every telemetry record. With
`--out`, `score` also writes the run's own `run_config.json` (the pinned one
with the service identity the run used) beside the summary. Attack
success is the benchmark's `security` field, where `true` means the injected
goal was achieved.

## Reproducing

[`reproduce.sh`](reproduce.sh) clones the fork at the pinned commit, installs
the bundled AgentDojo package, optimizes injections against each arm, runs the
benign and attacked benchmarks into a new run directory, reads the service's
build id at the start and the end, and scores the run with `score`. It makes paid
model calls and refuses to start unless `CSBENCH_CONFIRM_PAID_RUN=1`. It needs
`OPENAI_API_KEY`, `GEMINI_API_KEY` and `NEXT_MOCA_API_KEY`. The optimizer is
stochastic, so a fresh run produces different injections and attack success
is expected to vary.

## Credit

AutoDojo is the work of Xinhang Ma, Taoran Li, Chaowei Xiao, Zhiyuan Yu, Ning
Zhang and Yevgeniy Vorobeychik ([paper](https://arxiv.org/abs/2606.15057)),
built on AgentDojo and AgentDyn, and is MIT-licensed. The fork keeps the
upstream `LICENSE`.
