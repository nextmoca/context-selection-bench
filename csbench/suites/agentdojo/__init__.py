"""AgentDojo suite: paired prompt-injection evaluation with a selection boundary.

The suite runs AgentDojo (an agent-security benchmark with scored user tasks,
injection tasks and tool environments) as two matched arms:

- ``baseline``: AgentDojo's unmodified agent pipeline.
- ``needlepath_r6_text_autochunk``: the same pipeline with one element inserted
  after the tool executor. Whenever the newest message suffix is a contiguous
  batch of tool results, that batch (and nothing else) is submitted to the
  hosted selector through its public simple-text ingress, and the rendered
  selection replaces the batch only when the selector engaged. Every other
  path preserves the original tool messages byte for byte.

Three outcomes are measured: task utility, attack success (AgentDojo's
``security`` field, where ``true`` means the attack succeeded) and
*intermediate tool-result context reduction*, a deliberately narrow token
measure over the submitted tool-result batches only.

Layout:

- ``config``     typed run configuration and the ``RunManifest`` schema
- ``ids``        deterministic case and turn identifiers
- ``bridge``     the pipeline element (a thin transport; no selection logic)
- ``telemetry``  one metadata-only JSONL event per submitted batch
- ``provider``   Gemini API-key client factory for AgentDojo's Google LLM
- ``preflight``  neutral replacement for AgentDojo's non-scored preflight
- ``collect``    per-case rows, whole-arm validation gates
- ``runner``     sharded execution with whole-arm retry and archival
- ``summarize``  gated paired summary (machine-readable JSON + per-case JSONL)
- ``shard``      the per-shard entry point the runner launches

The ``agentdojo`` and ``google-genai`` packages are optional (``pip install
-e ".[agentdojo]"``); every module here imports without them, and the
regression tests run against fakes with no network.
"""

ARM_BASELINE = "baseline"
ARM_NEEDLEPATH = "needlepath_r6_text_autochunk"
ARMS = (ARM_BASELINE, ARM_NEEDLEPATH)

__all__ = ["ARM_BASELINE", "ARM_NEEDLEPATH", "ARMS"]
