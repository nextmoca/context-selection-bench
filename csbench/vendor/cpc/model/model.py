from typing import List, Optional, Tuple, Union
import torch

from transformers import (
    MistralModel,
    MistralPreTrainedModel,
    MistralForCausalLM,
    MistralConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
    MistralMLP,
)
from torch import nn
from transformers.utils import logging
from model.attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from training.loss import HardNegativeNLLLoss
import training.loss as losses_module

from dataclasses import dataclass

from peft import PeftModel
from peft import LoraConfig, get_peft_model

import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, StaticCache, SlidingWindowCache

from args import parse_args

from transformers import AutoTokenizer, AutoModel, AutoConfig
from util.torch_util import  mean_pooling, cos_sim

from model.common import (
    ensure_model_type,
    build_model,
    resize_model_embeddings_to_fit_tokenizer,
)


def load_model_and_tokenizer(config_path, tokenizer_name_or_path, lora_name_or_path):
    model_info = build_model(config_path, tokenizer_name_or_path)

    model = model_info.model
    tokenizer = model_info.tokenizer

    model = resize_model_embeddings_to_fit_tokenizer(model, tokenizer)

    # NOTE (needlepath vendoring): original code called `PeftModel.from_pretrained`
    # with no `torch_device`, so peft's `infer_device()` picked "mps" on
    # Apple Silicon (checked before "cpu"). torch 2.2's MPS backend cannot
    # materialize bfloat16 tensors, so loading the (bf16) LoRA adapter
    # checkpoint crashed with "Trying to convert BFloat16 to the MPS
    # backend...". Force the adapter checkpoint onto CPU when CUDA is
    # unavailable. See VENDOR.md.
    peft_torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model = PeftModel.from_pretrained(model.model, lora_name_or_path, torch_device=peft_torch_device)

    # NOTE (needlepath vendoring): original code unconditionally called
    # `model.cuda()`, which crashes on machines with no CUDA device (this
    # Mac runs CPU-only). Made device-aware: use CUDA when available,
    # otherwise stay on CPU. See VENDOR.md.
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to("cpu")

    # model.model = model.model.merge_and_unload()

    return model, tokenizer
