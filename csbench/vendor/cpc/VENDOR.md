# Vendored: Workday/cpc

- **Source**: https://github.com/Workday/cpc
- **Pinned commit**: `66b623fa4c3bbcd25469874b67614ccb4e2aa08b` (2025-01-16)
- **License**: Apache-2.0 (see `LICENSE` in this directory, copied verbatim from upstream)
- **Paper**: Liskavets et al., "Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference" (AAAI 2025), https://arxiv.org/abs/2409.01227

## What was vendored

Only what the inference path (`prompt_compressor.PromptCompressorCPC.compress()`)
needs:

- `prompt_compressor.py`
- `args.py` (imported by `model/common.py` for `parse_args`)
- `model/` (all files: `common.py`, `model.py`, `llama.py`, `mistral.py`,
  `qwen2.py`, `multi_lora.py`, `attn_mask_utils.py`, `__init__.py`)
- `util/` (all files: `preprocessing.py`, `util.py`, `torch_util.py`, `__init__.py`)
- `configs/` (all: `cpc-1.0-llama.json`, `cpc-1.0-mistral.json`, `prompts.json`)
- `training/loss.py` only: `model/llama.py`, `model/mistral.py`, and
  `model/model.py` do `from training.loss import HardNegativeNLLLoss`
  (used to instantiate `self.sim_loss` on the model class, even though the
  inference path never calls it). No other file under upstream
  `training/` (`callbacks.py`, `data_collator.py`, `train.py`) is imported by
  the inference path, so those were skipped.
- `LICENSE`, `README.md`

**Skipped** (not needed for inference): `images/`, `data_collection/`,
`evaluation/`, and the rest of `training/` (`callbacks.py`, `data_collator.py`,
`train.py`).

No `__pycache__`/`.pyc` files are vendored (upstream repo did not have any at
the pinned commit; none were introduced during vendoring).

## Import layout note

The upstream repo's code uses absolute top-level imports (`from model.common
import ...`, `from util.preprocessing import ...`, `from training.loss import
...`, `from args import parse_args`) that assume the repo root itself is on
`sys.path`, not that `model`/`util`/`training`/`args` are submodules of a
`cpc` package. The CPC arm reproduces this by inserting this directory onto
`sys.path` before importing `prompt_compressor`.

## Local modifications (all required to run on CPU-only, torch 2.2.2)

### 1. `configs/cpc-1.0-llama.json`: `attn_implementation` `flash_attention_2` -> `sdpa`

flash-attn is not installable on macOS. The upstream Llama-1B training
config hardcodes `"attn_implementation": "flash_attention_2"`; changed to
`"sdpa"` so the same config drives inference correctly on CPU. No other
field in this config file was changed.

### 2. `model/model.py`: `load_model_and_tokenizer`, device-aware `.cuda()` calls

- `model.cuda()` was unconditional; now only called if
  `torch.cuda.is_available()`, otherwise `model.to("cpu")`.
- `PeftModel.from_pretrained(model.model, lora_name_or_path)` did not pass a
  `torch_device`. peft's `load_peft_weights()` (via `infer_device()`)
  defaults to checking CUDA, then **MPS**, before falling back to CPU. On
  Apple Silicon that picked `"mps"` to materialize the (bfloat16)
  LoRA adapter checkpoint tensors, and torch 2.2's MPS backend cannot
  convert bfloat16 tensors, so loading crashed with `RuntimeError: ... While
  copying the parameter named "...lora_A...": ('Trying to convert BFloat16
  to the MPS backend but it does not have support for that dtype.',)`.
  Fixed by explicitly passing `torch_device="cuda" if
  torch.cuda.is_available() else "cpu"` to `PeftModel.from_pretrained`, so
  the adapter checkpoint is loaded on CPU (bypassing the CUDA/MPS auto
  device inference) rather than GPU/MPS auto-selection.

### 3. `util/preprocessing.py`: `compress_sample`, device-aware tensor placement

Three input tensors (`inputs_question['input_ids']`,
`inputs_question['attention_mask']`, and the three `inputs_context.*`
tensors) were hardcoded with `.cuda()`, which raises
`AssertionError: Torch not compiled with CUDA enabled` on a CPU-only
machine. Replaced with `.to(device)` where `device =
next(model.parameters()).device`, resolved once per call so the tensors
always match wherever the model actually lives (CPU here; CUDA on a GPU
box, unchanged behavior there).

### 4. `util/util.py`: lazy-load the `evaluate` accuracy metric

`metric = evaluate.load("accuracy")` ran at **import time** of
`util/util.py`, which triggers a network fetch of the metric builder
script on every import, even though `compute_metrics`/`preprocess_logits_for_metrics`
(the only consumers of `metric`) are training-only functions never called
by the inference-only `PromptCompressor.compress()` path this harness
exercises. Since only the inference path is vendored, made this lazy:
`evaluate` is now imported and `evaluate.load("accuracy")` called inside a
`_get_metric()` helper only when `compute_metrics` actually runs, so
importing this module for inference requires no network access after
packages are installed.

### 5. `util/util.py`: `tokenize_and_clip_segments`, accept `sentence_embedding_type`

**Real upstream bug**, unrelated to CPU/macOS: `util/preprocessing.py`'s
`SamplePreprocessor.__call__` calls
`tokenize_and_clip_segments(..., sentence_embedding_type=self.sentence_embedding_type,
...)`, but the function's signature at the pinned commit does not declare
that parameter; every call raised `TypeError:
tokenize_and_clip_segments() got an unexpected keyword argument
'sentence_embedding_type'`, i.e. `PromptCompressorCPC.compress()` was
unconditionally broken as vendored, independent of any platform
difference. Fixed by adding `sentence_embedding_type=None` to the
signature; the function body never references it, so it is accepted and
ignored (no behavior change beyond no longer crashing).

## Performance note: never run this model in bfloat16 on Apple Silicon CPU

The config's `"torch_dtype": "bfloat16"` is fine on a CUDA GPU, but on
Apple Silicon (torch 2.2.2, ARM CPU) bf16 matmuls fall back to slow
single-threaded kernels: one real ~8k-token RULER item did not finish a
single `compress()` in 45+ minutes at 100% of one core. Measured
alternatives on the same item (identical selected-sentence count, 331):

| device/dtype | compress() latency |
|---|---|
| CPU bfloat16 (config as loaded) | pathological, killed at 45+ min |
| CPU float32, `torch.set_num_threads(10)` | ~52 s |
| MPS float16 | ~14 s |

This is handled **outside the vendored code** (no vendored-source change):
the CPC arm casts the loaded model with
`compressor.model.to(device=..., dtype=...)` after construction (MPS ->
float16, CPU -> float32) and sets `torch.set_num_threads()`. Note torch
2.2's MPS backend cannot hold bf16 tensors at all (same constraint as
modification #2 above), so float16 is the MPS dtype. Also confirmed:
`compress_sample` runs one model forward **per chunk** (~3 chunks of
<=6144 tokens for an 8k item), not per sentence; there is no missing
batching knob; the dtype/threading was the whole problem.

## Dependencies required by the CPC arm environment

| package | version | why |
|---|---|---|
| `transformers` | `4.46.3` | required; CPC's attention classes were removed in 4.57+ (see note below) |
| `tokenizers` | `0.20.3` | pulled in automatically by `transformers==4.46.3` |
| `peft` | `0.19.1` | CPC wraps the base model with a LoRA adapter (`PeftModel`) |
| `sentencepiece` | `0.2.1` | tokenizer backend dependency |
| `munch` | `4.0.0` | `util/preprocessing.py`, `model/common.py` use `Munch` for attribute-style dicts |
| `sentence_splitter` (PyPI: `sentence-splitter`) | `1.4` | `util/util.py` sentence segmentation |
| `evaluate` | `0.4.6` | imported by `util/util.py` (now lazy, see modification #4 above) |

`torch` (2.2.2) and `tiktoken` (0.11.0) are also required.

### Transformers version: why 4.46.3 is required

CPC's `model/llama.py` and `model/mistral.py` subclass per-backend attention
implementation classes that transformers removed in a later refactor to a
unified attention interface: `LlamaFlashAttention2`, `LlamaSdpaAttention`,
`MistralFlashAttention2`, `MistralSdpaAttention` do not exist in
`transformers==4.57.6`'s `modeling_llama.py`/`modeling_mistral.py` anymore
(`ImportError: cannot import name 'LlamaFlashAttention2' ...`). These
classes still exist in `transformers==4.46.3`. No vendored-code patch was
attempted for this: rewriting CPC's custom bidirectional-attention model
classes against the new unified attention interface would be a large,
non-minimal patch, whereas pinning transformers to a version CPC's code was
actually written against is the narrower, no-source-changes fix.

`sentence-transformers==2.7.0` and `llmlingua==0.2.2` both import successfully
(including `from llmlingua import PromptCompressor`) under
`transformers==4.46.3`, so this pin is compatible with the other compression
arms. Re-verify compatibility before upgrading to `transformers` 4.57+, which
removes the per-backend attention classes CPC subclasses.

## Config used

`configs/cpc-1.0-llama.json` (Llama-1B, local CPU smoke only): base
`unsloth/Llama-3.2-1B-Instruct`, LoRA
`deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn`, tokenizer
`deadcode99/cpc-1.0-llama-1b-tokenizer`. All ungated on HF, no token needed.
Weights download to the default HF cache (`~/.cache/huggingface`), never
committed to git.
