# AgentDojo suite

A matched-pair harness that runs [AgentDojo](https://github.com/ethz-spylab/agentdojo)
with a hosted context-selection method placed at the last hop between agent
tool results and the next model call. It measures three outcomes per arm:

1. **Utility**: does the agent still complete its legitimate task?
2. **Attack success**: does an injected instruction still achieve its goal?
   AgentDojo stores this under a field named `security`, where `true` means the
   attack *succeeded*. The harness never uses the console's "Average security"
   label.
3. **Intermediate tool-result context reduction**: how much of the submitted
   tool-result context reaches the next model call. This is a deliberately
   narrow measure over the selected batches only; it excludes system prompts,
   user messages, tool schemas, prior assistant turns, output tokens and price,
   and it is never labelled as a total cost reduction.

The reference configuration is one suite (`workspace`), one attack family
(`important_instructions`), one answer model, 40 user tasks x 14 injection
tasks = 560 scored rows per arm, five planned repeats. It is not the full
AgentDojo matrix and is not a universal prompt-injection result.

## Arms

| arm | pipeline |
|---|---|
| `baseline` | AgentDojo's unmodified pipeline: system message, query, LLM, tool loop |
| `needlepath_r6_text_autochunk` | the same pipeline with one element inserted after the tool executor |

One variable changes per arm. Both arms share the AgentDojo version, suite,
attack set, model, task population, tool environment, sharding and pipeline
name (so the attack's model wording is identical).

### The bridge (`bridge.py`)

Whenever the newest message suffix is a contiguous batch of tool results, the
bridge:

1. collects that batch and nothing else (system, user, developer, policy,
   assistant and tool-schema content never cross the boundary);
2. builds a raw text candidate: one result unchanged, several joined with
   stable `tool result N:` labels;
3. submits it through the public simple-text ingress (`text` plus
   `budget.autochunk = true`) with a deterministic `request_id` built from
   suite, user task, injection task, arm, repeat and turn index;
4. applies the rendered selection only when the server engaged and returned
   non-empty rendered text: it replaces the first tool result, and every later
   tool message in the batch becomes a fixed omission marker, so message count
   and order are preserved;
5. otherwise (stand-down, fallback, escalation, unsafe verdict, empty or
   malformed response, timeout, any exception) hands back the original message
   objects untouched.

The bridge is a transport. It does not decide what to keep: the server renders
the selection and the bridge forwards it verbatim or passes the originals
through.

### Telemetry (`telemetry.py`)

One JSONL event per submitted batch, metadata only (no prompt text, tool text,
rendered context or credentials), with a closed field set:
`event`, `ingress`, `autochunk`, `operating_point`, `tool_results`,
`original_token_estimate`, `selected_token_estimate`, `latency_ms`,
`engine_latency_ms` (documented), `gate_reason` (the documented `gate.reason`),
`reason` (the bridge's own decision), `records_expanded` (documented),
`format_metrics` (an opaque passthrough of the documented free-form
`format_metrics` object, reduced to scalar label-shaped entries; no key is
named by the harness), `case_id`, `turn_index`, `request_id`. The token
estimator is `max(1, len(text) // 4)`: deterministic and sufficient for
within-run comparison, not provider-billed accounting.

## Protocol invariants

`validate_config` refuses a configuration outside the protocol: at most two
concurrent shards, at most three whole-arm attempts, one SDK attempt per model
call (the benchmark's own retry wrapper owns retries), a per-request model
timeout of at most 60 s, `autochunk = true`, `fallback_policy = fail_open`,
`entry_point = text`. A resumed run must present the exact configuration it
started with. The summarizer's planned repeat count is the manifest's and
cannot be overridden.

## Validity and retry

An arm is accepted only when every shard exited 0, no row carries a provider
error, and exactly the expected case ids exist (no duplicates, nothing
missing, nothing extra); the selector arm must additionally have telemetry in
which every event carries an operating point, ingress and token estimate. An
invalid arm is archived to `<arm>.failed_attempt_<timestamp>` and retried from
an empty directory, up to three attempts. Every attempt is recorded in the
manifest with its id, exit codes, row count and invalidation reason.

The summarizer refuses to produce output when any gate fails, when the
baseline and selector case sets differ in any repeat, when the manifest is
under-described, or when fewer than the planned repeats are complete. A
directional-only summary of at least two complete repeats can be requested
explicitly; it says so in its output and reports no confidence interval.

## Layout of a run root

```text
<run-root>/
  run_config.json
  manifest.json
  repeat_1/
    baseline/
      console-shard-0.log ... console-shard-7.log
      <pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json   (AgentDojo rows)
      timing.json
    needlepath_r6_text_autochunk/
      console-shard-*.log
      needlepath_metrics_shard_0.jsonl ... needlepath_metrics_shard_7.jsonl
      <AgentDojo rows>
      timing.json
    needlepath_r6_text_autochunk.failed_attempt_<timestamp>/   (archived, never read)
  summary/
    summary.json
    cases.jsonl
```

## Running

```bash
pip install -e ".[dev,agentdojo]"
export GEMINI_API_KEY=...          # or NM_GEMINI_API_KEY; never logged
export NEEDLEPATH_API_KEY=...

python -m csbench.suites.agentdojo init-config \
  --run-id workspace-ii-r6-5rep --out run_config.json \
  --selector-url https://<hosted-endpoint> --selector-snapshot <selector-commit-or-image>

python -m csbench.suites.agentdojo validate run_config.json
python -m csbench.suites.agentdojo run --config run_config.json --out runs/agentdojo/<run-id>
python -m csbench.suites.agentdojo summarize --run-root runs/agentdojo/<run-id>
```

`run` makes paid model calls (two shards concurrently per wave, eight shards
per arm). `summarize` and `validate` are offline.

## What is patched in AgentDojo

Exactly one thing: the non-scored preflight that runs each injection task as
a user task (`agentdojo.benchmark.run_task_without_injection_tasks`) is
replaced by a neutral function that makes no model call and writes no file,
in both arms. It races across parallel shards and costs a model call per
injection task otherwise. Its console line (`Passed injection tasks as user
tasks: 0/14`) is not a metric. The LLM element and the bridge are injected
through AgentDojo's own `PipelineConfig(llm=...)` and the pipeline element
list; nothing else is monkeypatched.

## Tests

`tests/test_agentdojo_*.py` run offline with fakes and cover the protocol's
regression list: baseline equivalence, applied selection, fail-open on
timeout / exception / empty / stand-down / malformed, context isolation, the
public-text contract, telemetry privacy, row completeness (559 of 560 fails
the arm), pair matching, retry isolation and model-alias recording. One
integration test exercises the real `AgentPipeline` with a scripted LLM when
the `agentdojo` extra is installed, and is skipped otherwise.

## Not yet done

- A live run. No numbers are shipped with this suite.
- Provider-reported input-token usage and a model-family tokenizer estimate
  alongside the heuristic estimator.
- Replacing the preflight patch with an upstream-supported switch.
- Upload of accepted artifacts with checksums, and per-attack breakdowns.
