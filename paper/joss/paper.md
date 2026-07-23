---
title: 'context-selection-bench: A matched-protocol harness for context-selection and prompt-compression methods'
tags:
  - Python
  - large language models
  - benchmarking
  - reproducibility
  - context selection
  - prompt compression
  - evaluation
authors:
  - name: Kiran Kashalkar
    orcid: 0009-0001-2002-593X
    corresponding: true
    affiliation: "1"
affiliations:
  - name: Next Moca Global, Inc.
    index: 1
date: 23 July 2026
bibliography: paper.bib
---

# Summary

Production large language model (LLM) systems rarely pass their full accumulated
state to the model. A preprocessing step first fits that state to a token
budget, either by context selection (keeping or dropping whole records) or by
prompt compression (rewriting text into fewer tokens). `context-selection-bench`
(csbench) is a reproducible benchmark harness that evaluates both families of
method under one matched protocol on public evaluation suites, so their effect
on task accuracy, token reduction, and latency can be compared directly rather
than across separately reported papers.

Every method under test is a uniform *arm* implementing one interface,
`select(ContextRequest) -> ContextResponse` (`csbench/arms/base.py`). The request
carries the records, the task, and a token budget; the response carries the
model-facing context together with neutral token, fallback, and latency metadata
(`csbench/contracts.py`). Local arms run in process (a full-context control and
the LLMLingua-2 [@pan2024llmlingua2] and CPC [@liskavets2025cpc] compressors); a
hosted method is reached over HTTP by a thin adapter that holds no selection
logic itself (`csbench/arms/needlepath.py`). Because every arm shares the
interface, the items, one model, and one scorer per suite, differences in the
results reflect the methods, not the measurement.

The harness bundles five suites, RULER [@hsieh2024ruler], SQuAD v2
[@rajpurkar2018squad2], BFCL [@bfcl2024], TruthfulQA [@lin2022truthfulqa], and
GSM8K [@cobbe2021gsm8k], loaded from public sources through the HuggingFace
`datasets` library [@lhoest2021datasets]. Each is graded by a single inspectable
scoring surface (`csbench/stats.py`, `csbench/suites/`) that implements the
suite's reference metric and applies it identically to every arm. On top of
scoring, csbench provides a paired-statistics module (McNemar's test
[@mcnemar1947] and a seeded, paired bootstrap confidence interval on an accuracy
delta) built on NumPy [@harris2020numpy] and SciPy [@virtanen2020scipy]; latency
measurement classes that label each figure and decompose a remote call into
transport and service time (`csbench/latency.py`); sha256 manifests over dataset
inputs and per-item outputs that any third party can recompute
(`csbench/provenance.py`, `tools/verify_manifests.py`); and a pre-run spend
estimate with an opt-in hard cap that aborts before any model call.

# Statement of need

Context selection and prompt compression are now standard in production LLM
pipelines, and many methods exist: token-level compressors such as LLMLingua and
LongLLMLingua [@jiang2023llmlingua; @jiang2024longllmlingua], task-agnostic
distillation such as LLMLingua-2 [@pan2024llmlingua2], self-information pruning
such as Selective Context [@li2023selectivecontext], sentence-level selection
such as CPC [@liskavets2025cpc], and agent-oriented compression such as ACON
[@acon2025]. Each is usually published with its own datasets, its own baseline,
and its own scoring code, which makes head-to-head claims difficult to reproduce
and easy to dispute.

General-purpose evaluation harnesses do not resolve this. lm-evaluation-harness
[@gao2023evalharness], HELM [@liang2023helm], and long-context benchmarks such as
LongBench [@bai2024longbench] evaluate a model on a task. None has a first-class
notion of a preprocessing layer sitting between the accumulated state and the
model, so none can place a selection method, a compression method, and a
full-context control on the same items under one protocol, with the same model
and the same scorer.

csbench is built for exactly that comparison, and adds the measurement
disciplines a fair comparison needs. It fixes the base model, the items, and the
scorer across arms; pairs items index-for-index so a paired significance test is
valid; reports a paired bootstrap confidence interval and McNemar's test in place
of a bare point estimate; pins inputs and outputs with recomputable sha256
manifests; labels every latency figure with a measurement class and separates
transport from service time; and bounds cost with a pre-run estimate and an
opt-in cap. No prior harness we are aware of combines these disciplines for
context-reduction methods, which is the gap this software fills for researchers
and practitioners who need defensible, reproducible comparisons.

# State of the field

The methods above ship as libraries or services, not as neutral comparison
apparatus. The LLMLingua family [@jiang2023llmlingua; @jiang2024longllmlingua;
@pan2024llmlingua2], Selective Context [@li2023selectivecontext], CPC
[@liskavets2025cpc], the open-source Headroom toolkit [@headroom], and the hosted
Compresr API [@compresr] each supply one method and evaluate it on surfaces their
authors choose. csbench treats each as a single arm and grades it with the
suite's scorer under the shared protocol, so a new method earns its comparison
rather than asserting it. Adding a method means implementing one `select()`
function or pointing the HTTP adapter at an endpoint.

The hosted-adapter path is the second distinctive contribution: it lets a closed,
served system be benchmarked under the same contract as open arms. The adapter
names a method's configuration only by an opaque, versioned, immutable
`operating_point` label, and records the resolved policy version with every item.
This makes a hosted method's published run verifiable (its per-item outputs match
a committed sha256 manifest) and re-runnable (cite the label, call the endpoint),
while being explicit that it is not inspectable (the mapping from label to
configuration is not disclosed). Local arms carry no such registry and are fully
open in the repository. A contract-conformance stub server (`csbench.stub`)
validates the wire schema in continuous integration without emulating any
method's behavior. The interface, protocol, and reproducibility disciplines are
documented in the repository (`README.md`, `INTERFACE.md`) with installation
instructions and runnable examples, and the code is released under Apache-2.0.

# Acknowledgements

Swanand Rao designed the Needlepath selection method [@rao2026needlepath] and the
research protocol that motivated this harness. This work was supported by
Next Moca Global, Inc.

# References
