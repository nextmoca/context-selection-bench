# Deposit: structure, per-run license scope, and how to rebuild it

This documents the Zenodo deposit that accompanies the paper. It is a
mixed-license package assembled deterministically by `tools/build_deposit.py`
from the per-run artifacts. The git repository never contains the per-item run
data: `/runs/` is gitignored (it lives in object storage and in this deposit).

## Per-run scope (what ships)

Every run ships a `manifest.sha256` regardless.

| Run(s) | Upstream | Per-item prompts | Ships |
|---|---|---|---|
| `ruler_v1`, `ruler_repl_v1`, `gate_ruler_v1` | RULER (PG essays; SQuAD/HotpotQA; synthetic) | no (hash) | completions, scores, token counts, metadata; `qa_1`/`qa_2` answers hashed; `sha256(prompt)` where a prompt is present |
| `appworld_v1`, `gate_appworld_v1`, `gate_probe_v1` | AppWorld | none | aggregates + paired-stats + content hashes only |
| `controls_v1` - TruthfulQA | TruthfulQA (Apache-2.0) | plaintext | upstream items + our outputs, with `NOTICE` + citation |
| `controls_v1` - BFCL | BFCL (Apache-2.0) | plaintext | upstream items + our outputs, with `NOTICE` + citation |
| `gsm8k_v1` (separate full-set run) | GSM8K (MIT) | no (hash) | full-set exact-match aggregate + manifest, hash-only; this figure is from a separate full-set run, not the `controls_v1` harness; regenerate per-item via the harness |
| `controls_v1` - SQuAD v2 | SQuAD v2 (CC BY-SA 4.0) | no (hash) | outputs + scores; passages hash-only (share-alike would conflict with CC BY 4.0) |

## Hard excludes (never in the deposit, by construction)

- Gate engine-internal statistics (any `signals` path: the relevance-gap /
  task-signature-drift signals).
- The AppWorld dataset canary.
- Prompt / context / haystack text (RULER, AppWorld).
- Host / filesystem paths and internal identifiers (redacted from aggregates).

`build_deposit.py` enforces these with a strict per-item field **allowlist**
(unknown fields are dropped, never shipped) plus hard-exclude patterns, and then
re-runs `scripts/scrub_check.py` over the assembled output: the build fails if
anything trips the gate.

## Rebuild / verify

```bash
# RULER-family runs (per-item outputs; validated on ruler_repl_v1):
python tools/build_deposit.py --run-type ruler --run-id ruler_repl_v1 --src runs/ruler_repl_v1 --out deposit

# AppWorld and control runs assemble from their source dirs at release pre-flight:
#   --run-type appworld        (aggregates + paired-stats + hashes)
#   --run-type controls_plain  (TruthfulQA / BFCL plaintext + outputs)
#   --run-type controls_hash   (SQuAD v2 outputs + hashed passages)
#   --run-type gsm8k_hash      (GSM8K full-set aggregate + manifest, hash-only)
#   --run-type gate_ruler      (RULER gate engage outcomes; signal values dropped)

# Verify any run against its manifest:
python tools/verify_manifests.py deposit/<run>/manifest.sha256 --base-dir deposit/<run>
```

## Reproduction provenance (regenerate upstream prompts locally, under upstream terms)

- **RULER**: NVIDIA/RULER pinned commit `38da79d79519ef87aa46ae804f838e1eab7f86d7`;
  seed 42; `num_samples` 100; 13 tasks; lengths {8192, 16384}; Gemini tokenizer
  (`gemini-3.1-pro-preview` `countTokens`). Fetch: `python scripts/fetch_ruler.py`.
- **Controls**: pinned Hugging Face revisions per suite (see the fetch scripts and
  `DATASETS.md`).
- **AppWorld**: reproduce via the upstream AppWorld harness (Trivedi et al.); this
  deposit ships aggregates + hashes only.
