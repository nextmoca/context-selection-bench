# AgentDyn suite (published GPT-4o run)

[AgentDyn](https://github.com/SaFo-Lab/AgentDyn) is an agent-security
benchmark built on AgentDojo, with open-ended user tasks in three scenarios
(shopping, github, dailylife) and injection tasks planted in the tool
environments. This directory pins the code that produced the published GPT-4o
result, carries that result, and shows how to run it again.

## Pinned code

| | |
|---|---|
| upstream | `SaFo-Lab/AgentDyn` at `5353cf7615b135cace8d07c8f12dac53a16b6db3` (MIT) |
| fork | [`nextmoca/agentdyn`](https://github.com/nextmoca/agentdyn), branch `next-moca` |
| pinned commit | `46f7314d09380e87e7dad26931863c8fe7f37881` |
| run commit | `25937a2c83d6ba06d73cd9bd8e93417c733daef4`, recorded in every run log; the parent of the pinned commit, which adds only a README section |
| defense element | `src/agentdojo/agent_pipeline/next_moca.py`, sha256 prefix `b9698486c20eb54d` |

The `next_moca` element sits after the tool executor. It sends the trailing
batch of tool results to the hosted service and passes on the context the
service returns; on a timeout, an HTTP error, a malformed response or a server
pass-through directive the model sees the original tool output. It does not
score, rank or rewrite tool output itself.

## Run configuration

Full detail in [`run_config.json`](run_config.json).

- Model: `gpt-4o-2024-08-06` (OpenAI) in both arms.
- Suites: shopping (20 user tasks x 9 injection tasks), github (20 x 9),
  dailylife (20 x 10); benchmark version `v1.2.2`; attack
  `important_instructions`.
- Population per arm: 560 attacked tasks and 60 benign tasks, each once.
- Arms: `none` (AgentDyn's pipeline) and `next_moca`.
- Service: Next Moca hosted service, November 2026, build `42aea840d7689d36`
  (the same build id at the start and the end of the run). The operating
  point is not a listed label and is not published.
- Dates: both arms ran on 2026-11-10 (UTC), as separate runs from the same
  fork commit.

## Results

[`results/summary.json`](results/summary.json). Counts are tasks.

| suite | attacked tasks | attack success, none → next_moca | tasks completed under attack, none → next_moca | benign tasks completed, none → next_moca |
|---|---|---|---|---|
| dailylife | 200 | 130 → 28 | 104 (52.0%) → 113 (56.5%) | 12/20 → 14/20 |
| github | 180 | 34 → 6 | 129 (71.7%) → 125 (69.4%) | 13/20 → 14/20 |
| shopping | 180 | 53 → 1 | 83 (46.1%) → 84 (46.7%) | 11/20 → 9/20 |
| pooled | 560 | 217 → 35 | 316 (56.4%) → 322 (57.5%) | 36/60 → 37/60 |

Per-row result files for this run are not included in this release; the
summary records the per-suite counts the scorer produced from them.

## Checking and re-scoring

```bash
pip install -e ".[dev]"
python -m csbench.suites.agentdyn check     # offline consistency check of run_config.json and results/summary.json
python -m csbench.suites.agentdyn score --none <logdir> --next-moca <logdir> --build-id <id> --out <dir>/summary.json
```

For this suite `check` can confirm only that the committed summary matches the
pinned configuration, the planned population and its own arithmetic: without
the per-row files it cannot re-derive the counts. `score` reads
AgentDojo-format result rows under each arm's log directory and refuses to
produce a summary when a row is incomplete, unreadable, not boolean-scored or
marked with a provider error; when a case appears twice; when a suite is not
the full planned user-task by injection-task population; when the two arms did
not score the same cases; or when the directory given for an arm does not hold
that arm's rows (the defended arm must carry the element's telemetry, the
other must not). `--build-id` is the build id the service reported for the
run. Attack
success is the benchmark's `security` field, where `true` means the injected
goal was achieved.

## Reproducing

[`reproduce.sh`](reproduce.sh) clones the fork at the pinned commit, installs
it in a fresh virtual environment, runs both arms with AgentDyn's own
benchmark script into a new run directory, reads the service's build id at the
start and the end, and scores the run with `score`. It makes paid model calls
and refuses to start unless `CSBENCH_CONFIRM_PAID_RUN=1`. It needs
`OPENAI_API_KEY`, `NEXT_MOCA_API_KEY` and `NEXT_MOCA_OPERATING_POINT`. The
published arm used an operating point that is not a listed label, so a
reproduction runs on a listed operating point of your choice and is a related
measurement, not a re-run of the published arm.

## Credit

AgentDyn is the work of Hao Li, Ruoyao Wen, Shanghao Shi, Ning Zhang,
Yevgeniy Vorobeychik and Chaowei Xiao ([paper](https://arxiv.org/abs/2602.03117)),
built on AgentDojo, and is MIT-licensed. The fork keeps the upstream `LICENSE`.
