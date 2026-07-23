# Arm interface

Every method under test, local or hosted, implements one contract:

```
select(ContextRequest) -> ContextResponse
```

Local arms (full-context passthrough, and in-process compression baselines)
implement it directly. Hosted methods are reached over HTTP by a thin client
(`csbench.arms.NeedlepathArm`); the method runs on an endpoint the provider
operates, and only this contract crosses the boundary.

This document is the wire contract. It is deliberately explicit about what a
hosted method does **not** expose, so a skeptical reader can see exactly what is
reproducible and what is verifiable.

## Endpoint

```
POST {base_url}/v1/context/select
Content-Type: application/json
Authorization: Bearer <api_key>      # when the endpoint requires it
```

Request body = `ContextRequest.to_wire()`. Response body = `ContextResponse.to_wire()`.

## Request: `ContextRequest`

| field | type | notes |
|---|---|---|
| `request_id` | str | per-unit id; the response must echo it. |
| `records` | `ContextRecord[]` | the state to keep/drop/compress. |
| `task` | `TaskSpec` | the query the context must serve. |
| `budget` | `BudgetSpec` | the operating point (below). |
| `render` | bool | ask the server for a model-facing string. |
| `render_format` | `"plain" \| "hybrid"` | neutral render hint. |
| `return_per_record` | bool | include `selected[]` detail. |

`ContextRecord`: `text` (verbatim content), `kind` (role enum:
`user_input`/`llm_response`/`tool_call`/`tool_result`/`external_data`/`error`/`artifact`/`tool_schema`),
`id?`, `source?`, `title?`, `step_id?`, `importance` (relevance prior),
`keywords[]`, `tags[]`, `attributes{}` (opaque per-kind passthrough; derived
quantities such as token counts are computed server-side and are not sent).

`TaskSpec`: `prompt`, `tool_name?`, `required_record_ids[]`,
`parent_record_ids[]`, `keywords[]`, `tags[]`, `step_id?`, `recent_prompts[]`
(prior-step prompts, for drift signals), `output_mode?`, `output_token_budget?`.

`BudgetSpec`: `max_context_tokens`, `operating_point?` (see registry),
`max_records?`, `max_excerpt_tokens_per_record?`, `mode` (`"fixed" | "adaptive"`),
`adaptive?` (`{initial_tokens, escalation_tokens[], allow_full_context_fallback}`),
`require_evidence_coverage?`.

**No method-tuning knobs appear in the request.** Thresholds, weightings, tier
rules, gate parameters and the like are resolved server-side from the
`operating_point` label. A competitor arm never sees them, and they never cross
the boundary.

## Response: `ContextResponse`

Required fields (validated by the client and the stub): `request_id`,
`rendered_context`, `tokens_before`, `tokens_after`, `tokens_saved`,
`records_available`, `records_selected`, `fallback_used`, `engine_latency_ms`.

Full shape: `request_id`, `policy_version` (below), `rendered_context` (the
model-facing block), `selected[]` (`record_id`, `kind`, `title`, `source`,
`score`, `reason`, `excerpt`, `excerpt_format`, `selected_tokens`),
`tokens_before/after/saved`, `records_available/selected`, `fallback_used`,
`selection_error?`, `engine_latency_ms` (server-measured), `budget_tokens`,
`attempted_budget_tokens[]`, `reduction_ratio`, `safety?`, `gate?`,
`format_metrics{}`.

`safety` (neutral subset of a coverage/answerability verdict, a method's
internal obligation taxonomy stays server-side): `selection_safe`,
`fallback_required`, `fallback_reason`, `coverage_score`, `evidence_shape`,
`evidence_terms?`, `repair_reasons[]`.

`gate` (neutral gate summary): `engaged`, `reason`, `signals{}` (JSON-safe
telemetry).

The gate outcome is an **open, extensible enum**, not a fixed binary. Today it
takes two values: a *select* outcome (`engaged = true`) and a *stand_down*
outcome (`engaged = false`), with the specific trigger carried in `reason`
(e.g. `engage:needle`, `standdown:high_drift`). A third outcome (*escalate*,
signalled when a caller declares a hard downstream capacity cap and neither a
sufficient within-cap selection nor full pass-through is available) is a
planned addition. **Clients must treat the gate outcome and `reason` as open:**
tolerate unrecognized `reason` prefixes and do not assume the outcome is
strictly binary. When *escalate* ships it will be introduced additively (a new
`reason` prefix and/or an optional outcome field), so existing clients that
follow this rule will not break.

Competitor arms may leave `selected[]` empty and `safety`/`gate` null and rely
on `rendered_context` + token metadata; the row-writer never assumes
`selected[]` is populated.

## Operating-point registry

A hosted method's configuration is named by an **opaque, versioned, immutable**
`operating_point` label, for example `np-2026-07-r1`. The label maps,
server-side, to a frozen configuration. The rules:

- **Immutable.** A published label never changes meaning. Re-tuning produces a
  *new* label (`np-2026-07-r2`, ...); it never edits an existing one.
- **Pinned in results.** Every published run cites the exact label it used, and
  the provider commits to keeping that label re-runnable.
- **Echoed per item.** The response returns `policy_version` (the resolved,
  frozen version behind the label) and the harness records it with every item,
  so a result file is self-describing.

### The trade-off, stated plainly

This design is honest about a limitation, so a reviewer does not have to infer
it:

- Published results **are verifiable**: every per-item output is committed with
  a sha256 manifest, and anyone can recompute the manifest over the released
  outputs (`tools/verify_manifests.py`).
- Published results **are re-runnable**: cite the `operating_point` label, call
  the hosted endpoint, and reproduce the numbers.
- Published results **are not inspectable**: the mapping from label to
  configuration, and the method's internal logic, are not disclosed. You can
  confirm *what* was produced and reproduce it; you cannot read *how*.

Local competitor arms carry no operating-point registry: they are fully open
and inspectable in this repository.

## Timing

The harness times each arm's `select()` end-to-end with a monotonic clock and
counts before/after context sizes with the shared token heuristic
(`csbench.tokenizing.estimate_tokens`), identically for every arm. A hosted
method additionally reports its own `engine_latency_ms` (server-measured
selection time); the client records its round-trip time under
`format_metrics.client_latency_ms`. Headline arm timing uses the harness's
uniform wall-clock measurement so arms stay comparable.

### Measurement classes (`csbench.latency.MeasurementClass`)

A latency figure is meaningless without saying what baseline it was measured
against. Every published latency number carries exactly one class, and figures
of different classes are **never** compared without their labels:

| class | what it measures | which arms |
|---|---|---|
| `in_process` | pure local compute on our hardware, no network | local OSS compressors (CPC, LLMLingua-2) |
| `hosted_adapter_localhost` | request to a service we run on 127.0.0.1; loopback only | Needlepath, via the hosted adapter in these runs |
| `remote_api_e2e` | wall-clock round trip to a third-party API over the public internet, transport included | any remote commercial API (e.g. Compresr's `format_metrics.client_latency_ms`, tagged `client_latency_ms_class`) |
| `remote_api_network_adjusted` | `remote_api_e2e` minus the median TCP+TLS transport baseline to that host, an ESTIMATE of service-side time that still includes payload transfer and deliberately favors the vendor | derived, never raw |

A remote arm's `engine_latency_ms` may be the vendor's **own** server-side timer
(e.g. Compresr's `duration_ms`, also kept verbatim as
`format_metrics.vendor_reported_duration_ms`); that is vendor-self-reported, not
one of our four classes, and is used only as a cross-check.

`csbench.latency` provides the generic decomposition used for any remote arm:
`transport_baseline(host)` (cost-free TCP+TLS handshakes) and
`network_adjusted(e2e, transport_median)`. The CLI
`python -m tools.measure_remote_latency --host <api-host> [--arm <name>]`
produces the transport baseline and, optionally, a minimal-payload service floor.

## Errors

Non-2xx responses raise. A 2xx body missing any required field, or whose
`request_id` does not echo the request, raises `ContractError`.

## Stub server

The repository ships a contract-conformance stub (`csbench.stub`,
`python -m csbench.stub`). It validates requests and returns schema-valid
responses, but it is **not** a behavioral emulation: it passes context through
unchanged and emits no gate/safety verdict. It exercises wiring and schemas in
tests/CI; it does not produce benchmark results. See `csbench/stub/README.md`.
