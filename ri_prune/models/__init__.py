"""Model loading and adapter dispatch."""
from __future__ import annotations

from typing import Any

import torch

from .base import GenerationOutput, PreparedInputs, VLMAdapter
from .llava_ov import LlavaOneVisionAdapter
from .qwen_vl import QwenVLAdapter

ADAPTERS: tuple[type[VLMAdapter], ...] = (QwenVLAdapter, LlavaOneVisionAdapter)

# Short names accepted by the CLI.
MODEL_ALIASES = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}


def get_adapter(model: Any, processor: Any, query_scope: str = "text") -> VLMAdapter:
    model_type = model.config.model_type
    for cls in ADAPTERS:
        if model_type in cls.model_types:
            return cls(model, processor, query_scope=query_scope)
    supported = sorted(t for cls in ADAPTERS for t in cls.model_types)
    raise ValueError(f"Unsupported model_type {model_type!r}; supported: {supported}")


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device_map: str | dict | None = "auto",
    cache_dir: str | None = None,
    query_scope: str = "text",
) -> VLMAdapter:
    """Load a supported LVLM and wrap it in its adapter."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODEL_ALIASES.get(model_id, model_id)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    model.eval()
    return get_adapter(model, processor, query_scope=query_scope)


__all__ = [
    "ADAPTERS", "MODEL_ALIASES", "GenerationOutput", "PreparedInputs", "VLMAdapter",
    "LlavaOneVisionAdapter", "QwenVLAdapter", "get_adapter", "load_model",
]
