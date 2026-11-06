from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from transformers import DynamicCache


@dataclass
class PreparedInputs:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    visual_positions: torch.Tensor
    image_token_counts: list[int]
    query_positions: torch.Tensor
    position_ids: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_visual_tokens(self) -> int:
        return int(self.visual_positions.numel())

    @property
    def visual_tokens(self) -> torch.Tensor:
        return self.inputs_embeds[0, self.visual_positions]

    @property
    def query_embeds(self) -> torch.Tensor:
        return self.inputs_embeds[0, self.query_positions]


@dataclass
class GenerationOutput:
    text: str
    num_visual_tokens: int
    num_kept_visual_tokens: int
    prefill_length: int
    num_generated_tokens: int
    latency_generate_sec: float


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class VLMAdapter(ABC):
    model_types: tuple[str, ...] = ()

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.device = next(model.parameters()).device
        self.eos_token_ids: set[int] = set()
        for src in (getattr(model, "generation_config", None), self.tokenizer):
            eos = getattr(src, "eos_token_id", None)
            if eos is not None:
                self.eos_token_ids.update(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))

    @property
    def language_model(self) -> Any:
        return self.model.model.language_model

    @property
    def lm_head(self) -> Any:
        return self.model.lm_head

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def build_prompt(self, images: list[Image.Image], text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @abstractmethod
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs: ...

    @abstractmethod
    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor: ...

    def prefill_kwargs(self, prepared: PreparedInputs, keep_mask: torch.Tensor, keep_local: torch.Tensor) -> dict:
        return {}

    def _finish(self, input_ids, embeds, image_embeds, counts, image_token_id, **kw) -> PreparedInputs:
        visual_mask = input_ids[0] == image_token_id
        visual_positions = visual_mask.nonzero().flatten()
        if visual_positions.numel() != image_embeds.shape[0]:
            raise RuntimeError(f"{visual_positions.numel()} image tokens but {image_embeds.shape[0]} features")
        embeds[0, visual_positions] = image_embeds.to(embeds.dtype)
        return PreparedInputs(input_ids=input_ids, inputs_embeds=embeds, visual_positions=visual_positions,
                              image_token_counts=counts, query_positions=(~visual_mask).nonzero().flatten(), **kw)

    def keep_mask(self, prepared: PreparedInputs, keep_local: torch.Tensor | None) -> torch.Tensor:
        mask = torch.ones(prepared.input_ids.shape[1], dtype=torch.bool, device=prepared.inputs_embeds.device)
        if keep_local is not None:
            mask[prepared.visual_positions] = False
            mask[prepared.visual_positions[keep_local.to(mask.device)]] = True
        return mask

    @torch.no_grad()
    def generate(self, prepared: PreparedInputs, keep_local: torch.Tensor | None = None,
                 max_new_tokens: int = 32) -> GenerationOutput:
        device = prepared.inputs_embeds.device
        if keep_local is not None:
            keep_local = keep_local.to(device).long().sort().values
        mask = self.keep_mask(prepared, keep_local)
        local = keep_local if keep_local is not None else torch.arange(prepared.num_visual_tokens, device=device)
        embeds = prepared.inputs_embeds[:, mask]
        S = embeds.shape[1]

        sync(device)
        t0 = time.perf_counter()
        out = self.language_model(
            inputs_embeds=embeds, position_ids=self.prefill_position_ids(prepared, mask),
            attention_mask=torch.ones(1, S, dtype=torch.long, device=device),
            past_key_values=DynamicCache(), use_cache=True, return_dict=True,
            **self.prefill_kwargs(prepared, mask, local),
        )
        generated: list[int] = []
        for step in range(max_new_tokens):
            nxt = self.lm_head(out.last_hidden_state[:, -1]).argmax(dim=-1)
            tok = int(nxt.item())
            if tok in self.eos_token_ids:
                break
            generated.append(tok)
            if step == max_new_tokens - 1:
                break
            out = self.language_model(
                inputs_embeds=self.embed(nxt[:, None]),
                position_ids=self.decode_position_ids(prepared, S, step),
                attention_mask=torch.ones(1, S + step + 1, dtype=torch.long, device=device),
                past_key_values=out.past_key_values, use_cache=True, return_dict=True,
            )
        sync(device)

        return GenerationOutput(
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            num_visual_tokens=prepared.num_visual_tokens,
            num_kept_visual_tokens=int(local.numel()),
            prefill_length=S,
            num_generated_tokens=len(generated),
            latency_generate_sec=time.perf_counter() - t0,
        )
