from __future__ import annotations

import torch

from .base import GenerationOutput, PreparedInputs, VLMAdapter
from .llava_ov import LlavaOneVisionAdapter
from .qwen_vl import QwenVLAdapter

ADAPTERS = (QwenVLAdapter, LlavaOneVisionAdapter)
MODEL_ALIASES = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}


def get_adapter(model, processor) -> VLMAdapter:
    for cls in ADAPTERS:
        if model.config.model_type in cls.model_types:
            return cls(model, processor)
    raise ValueError(f"unsupported model_type {model.config.model_type!r}")


def load_model(model_id: str, dtype=torch.bfloat16, attn_implementation: str = "sdpa",
               device_map="auto", cache_dir: str | None = None) -> VLMAdapter:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODEL_ALIASES.get(model_id, model_id)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation=attn_implementation, device_map=device_map, cache_dir=cache_dir,
    ).eval()
    return get_adapter(model, processor)


__all__ = ["GenerationOutput", "PreparedInputs", "VLMAdapter", "QwenVLAdapter", "LlavaOneVisionAdapter",
           "get_adapter", "load_model"]
