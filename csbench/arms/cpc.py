"""CPC (Compress-then-Prompt Context compressor, AAAI'25) as a local arm.

A generic, in-process context-reduction arm wrapping the vendored CPC
compressor (``csbench.vendor.cpc``). Unlike the hosted Needlepath arm, all
selection runs locally: ``select`` concatenates the request's records into a
single context, hands that context plus the task prompt (as the question) to
CPC's real ``compress(...)`` loop, and returns the compressed text as the
rendered context.

CPC is a *selection-family* compressor: it greedily adds whole context
sentences by descending relevance until the running token count exceeds the
target, so every sentence in the output is a verbatim substring of the input
(it selects sentences, it never rewrites them).

Building the compressor performs a REAL model load (HuggingFace weight
download + torch inference prep), so ``torch``/``transformers`` and the
vendored model code are lazy-imported inside ``build_cpc_compressor`` and are
OPTIONAL dependencies: importing this module does not require them. Only the
lightweight ``tiktoken`` length measure (the basis CPC's own selection loop
targets, and therefore the basis the reduction figures are computed on) is
imported eagerly.

Device/dtype defaults are measured, not arbitrary: bfloat16 matmuls on ARM
CPUs fall back to slow single-threaded kernels, and some MPS backends cannot
represent bfloat16 at all, so CPU casts to float32 and MPS to float16 while
CUDA keeps the config's native bfloat16.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import tiktoken

from ..contracts import ContextRequest, ContextResponse
from .base import ContextArm

# The vendored CPC code (``csbench/vendor/cpc/``) uses top-level absolute
# imports (``from model.common import ...``, ``from prompt_compressor import
# ...``) exactly as they exist in the upstream repo layout. That directory
# itself, not this package, must be on ``sys.path`` for those imports to
# resolve. Insert it once at module import so any caller of
# ``build_cpc_compressor`` gets it regardless of import order.
_VENDOR_CPC_DIR = Path(__file__).resolve().parents[1] / "vendor" / "cpc"
if str(_VENDOR_CPC_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_CPC_DIR))

# CPC's LongBench operating point (the paper's fixed compression target). Kept
# as a reference default a caller may pass explicitly; by default the arm
# compresses to the request's own budget instead (see ``CpcArm.select``).
DEFAULT_COMPRESSION_TARGET_TOKENS = 2000

# device -> torch dtype name the model is cast to before inference (measured;
# see the module docstring for why bfloat16 is avoided off CUDA).
_DEFAULT_DTYPE_NAME_BY_DEVICE = {
    "cuda": "bfloat16",
    "mps": "float16",
    "cpu": "float32",
}

_MODEL_TYPE_NAMES = ("llama", "mistral")

# Default attention backend for every real CPC model build. ``sdpa`` is
# PyTorch's native ``scaled_dot_product_attention``: it needs no extra
# package (unlike ``flash_attention_2``, which requires a compiled wheel that
# is not installed on typical GPU runners), is numerically equivalent to
# flash-attention for inference, and works across CUDA, CPU, and MPS alike.
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"


@contextmanager
def _attn_implementation_override(attn_implementation: str) -> Iterator[None]:
    """Force every real CPC model build's ``attn_implementation``, regardless
    of what a vendored ``configs/cpc-1.0-*.json`` file says.

    The vendored Mistral config is kept byte-identical to upstream and still
    hardcodes ``flash_attention_2``; on a box without the ``flash_attn``
    package installed, ``transformers`` raises an ImportError. There is no
    constructor parameter to pass an override in: the config path is looked up
    from a hardcoded dict inside ``PromptCompressorCPC.__init__`` and read
    straight off disk. So instead of editing the vendored config file (a
    silent, easy-to-forget file-based default), this monkeypatches
    ``model.common.parse_args``, the single choke point every model build
    passes through, for the duration of one ``build_cpc_compressor()`` call,
    then restores the original. ``model/common.py`` does ``from args import
    parse_args``, binding the name into its own namespace, so the patch target
    is ``model.common.parse_args``, not ``args.parse_args``.
    """
    import model.common as _cpc_model_common  # vendor/cpc/model/common.py

    original_parse_args = _cpc_model_common.parse_args

    def _parse_args_with_override(config_path):
        model_args, data_args, training_args, custom_args = original_parse_args(config_path)
        model_args.attn_implementation = attn_implementation
        return model_args, data_args, training_args, custom_args

    _cpc_model_common.parse_args = _parse_args_with_override
    try:
        yield
    finally:
        _cpc_model_common.parse_args = original_parse_args


def resolve_device(requested: str, *, cuda_available: bool, mps_available: bool) -> str:
    """Pick the inference device. Explicit ``cpu``/``mps``/``cuda`` are
    respected as-is; ``auto`` prefers CUDA, then MPS, else CPU."""
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    return requested


def default_dtype_name_for_device(device: str) -> str:
    try:
        return _DEFAULT_DTYPE_NAME_BY_DEVICE[device]
    except KeyError as exc:
        raise ValueError(
            f"Unknown device {device!r}; expected one of {sorted(_DEFAULT_DTYPE_NAME_BY_DEVICE)}"
        ) from exc


def build_cpc_compressor(
    model_type: str = "mistral",
    *,
    device: str = "auto",
    dtype: Optional[str] = None,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
) -> Any:
    """Build and prepare a real CPC ``PromptCompressorCPC`` for local
    inference. ``model_type`` selects the Llama-1B config (light local runs) or
    the Mistral-7B config (the paper's operating point); ``device``/``dtype``
    follow the measured device-aware dtype casting described in the module
    docstring.

    ``attn_implementation`` (default ``"sdpa"``) is applied as a hard override
    on top of whatever the vendored config says: see
    ``_attn_implementation_override`` for the failure this prevents.

    ``torch``, ``transformers``, and the vendored model code are imported here
    (not at module import) because they are OPTIONAL, heavyweight dependencies.
    """
    import torch

    from model.common import ModelType  # vendor/cpc/model/common.py
    from prompt_compressor import PromptCompressorCPC  # vendor/cpc/prompt_compressor.py

    model_type_key = model_type.lower()
    if model_type_key not in _MODEL_TYPE_NAMES:
        raise ValueError(
            f"Unknown CPC model_type {model_type!r}; expected one of {_MODEL_TYPE_NAMES}"
        )
    model_type_enum = ModelType.LLAMA if model_type_key == "llama" else ModelType.MISTRAL

    resolved_device = resolve_device(
        device,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )
    resolved_dtype_name = dtype or default_dtype_name_for_device(resolved_device)
    resolved_dtype = getattr(torch, resolved_dtype_name)

    torch.set_num_threads(os.cpu_count() or 4)

    with _attn_implementation_override(attn_implementation):
        compressor = PromptCompressorCPC(
            model_type_enum, use_openai_tokenizer_to_measure_length=True
        )
    compressor.model = compressor.model.to(device=resolved_device, dtype=resolved_dtype)
    return compressor


_TOKEN_ENCODING = None


def _token_encoding():
    global _TOKEN_ENCODING
    if _TOKEN_ENCODING is None:
        _TOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4")
    return _TOKEN_ENCODING


def count_tokens(text: str) -> int:
    """Token count via the same tiktoken gpt-4 / cl100k encoding CPC itself
    uses to measure length against ``compression_target_tokens``, so the
    reduction figures are measured on the same basis CPC's selection loop
    targets."""
    return len(_token_encoding().encode(text or ""))


def concat_records(request: ContextRequest) -> str:
    """Join every record's text in document order into the single context CPC
    compresses. Records ARE the context here (there is no RULER-style
    context/question split to perform - the split is already reflected in the
    request: records are the context, ``task.prompt`` is the question)."""
    return "\n\n".join(rec.text for rec in request.records)


class CpcArm(ContextArm):
    """Local CPC compressor arm.

    ``select`` concatenates the request's records into a context, compresses it
    to ``request.budget.max_context_tokens`` with CPC's real ``compress(...)``
    (mirroring what the internal ``cpc_prompt`` did to the context portion),
    and returns the compressed text as the rendered context. The compressor is
    built lazily on first use (a heavyweight model load) unless one is injected.
    """

    name = "cpc"

    def __init__(
        self,
        *,
        model_type: str = "mistral",
        device: str = "auto",
        dtype: Optional[str] = None,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        target_tokens: Optional[int] = None,
        compressor: Any = None,
    ) -> None:
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        # When None, ``select`` compresses to the request's own budget; set an
        # explicit value (e.g. DEFAULT_COMPRESSION_TARGET_TOKENS) to pin a
        # fixed target across requests.
        self.target_tokens = target_tokens
        self._compressor = compressor

    def _get_compressor(self) -> Any:
        if self._compressor is None:
            self._compressor = build_cpc_compressor(
                self.model_type,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
        return self._compressor

    def select(self, request: ContextRequest) -> ContextResponse:
        t0 = time.perf_counter()

        context = concat_records(request)
        question = request.task.prompt
        target_tokens = (
            self.target_tokens
            if self.target_tokens is not None
            else request.budget.max_context_tokens
        )

        tokens_before = count_tokens(context)

        if context.strip():
            compressor = self._get_compressor()
            compressed = compressor.compress(
                context=context,
                question=question,
                compression_target_tokens=target_tokens,
            )
        else:
            compressed = ""

        tokens_after = count_tokens(compressed)
        tokens_saved = tokens_before - tokens_after
        reduction_ratio = (tokens_saved / tokens_before) if tokens_before else 0.0
        latency_ms = (time.perf_counter() - t0) * 1000.0

        records_available = len(request.records)
        # CPC selects whole sentences across the merged context rather than at
        # record granularity, so there is no faithful per-record excerpt to
        # emit; ``selected`` is left empty and ``records_selected`` reflects the
        # records fed into compression.
        records_selected = records_available if compressed.strip() else 0

        return ContextResponse(
            request_id=request.request_id,
            rendered_context=compressed,
            selected=[],
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            records_available=records_available,
            records_selected=records_selected,
            fallback_used=False,
            engine_latency_ms=latency_ms,
            budget_tokens=target_tokens,
            reduction_ratio=reduction_ratio,
            safety=None,
            gate=None,
        )
