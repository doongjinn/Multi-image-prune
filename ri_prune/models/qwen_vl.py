"""Qwen2.5-VL and Qwen3-VL adapters.

Both use M-RoPE: every token has a (temporal, height, width) position triple computed by
`get_rope_index` on the full prompt. Kept tokens keep their original triples, and generated
tokens continue at max(position) + 1, exactly as in unpruned decoding.

Qwen3-VL additionally injects "deepstack" visual features (from intermediate ViT layers)
into the first decoder layers at the visual positions. Those features are pruned with the
same indices as the projector outputs, so every kept token gets its own deepstack features.
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import PreparedInputs, VLMAdapter


class QwenVLAdapter(VLMAdapter):
    model_types = ("qwen2_5_vl", "qwen3_vl")

    def __init__(self, model: Any, processor: Any, query_scope: str = "text") -> None:
        super().__init__(model, processor, query_scope)
        self.is_qwen3 = model.config.model_type == "qwen3_vl"
        self.image_token_id = int(model.config.image_token_id)

    @property
    def language_model(self) -> Any:
        return self.model.model.language_model

    def build_prompt(self, images: list[Image.Image], text: str) -> str:
        """All images first, then the question (one user turn)."""
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}],
        }]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs:
        inner = self.model.model
        prompt = self.build_prompt(images, text)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        pixel_values = inputs.pixel_values.to(self.device)
        grid_thw = inputs.image_grid_thw.to(self.device)

        embeds = self.embed(input_ids)
        if self.is_qwen3:
            per_image, deepstack = inner.get_image_features(pixel_values, grid_thw)
        else:
            per_image, deepstack = inner.get_image_features(pixel_values, grid_thw), None
        counts = [int(f.shape[0]) for f in per_image]
        image_embeds = torch.cat(list(per_image), dim=0).to(embeds.dtype)

        visual_mask = input_ids[0] == self.image_token_id
        visual_positions = visual_mask.nonzero().flatten()
        if visual_positions.numel() != image_embeds.shape[0]:
            raise RuntimeError(
                f"{visual_positions.numel()} image tokens in the prompt but {image_embeds.shape[0]} features"
            )
        embeds[0, visual_positions] = image_embeds

        position_ids, _ = inner.get_rope_index(
            input_ids=input_ids, image_grid_thw=grid_thw, attention_mask=attention_mask,
        )
        extra: dict[str, Any] = {"max_position": int(position_ids.max().item())}
        if deepstack is not None:
            extra["deepstack"] = [d.to(embeds.dtype) for d in deepstack]

        return PreparedInputs(
            input_ids=input_ids,
            inputs_embeds=embeds,
            visual_positions=visual_positions,
            image_token_counts=counts,
            query_positions=self._query_positions(input_ids[0], visual_mask, text),
            position_ids=position_ids,
            extra=extra,
        )

    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor:
        return prepared.position_ids[:, :, keep_mask]

    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor:
        p = prepared.extra["max_position"] + 1 + step
        return torch.full((3, 1, 1), p, dtype=prepared.position_ids.dtype, device=prepared.position_ids.device)

    def prefill_kwargs(self, prepared: PreparedInputs, keep_mask: torch.Tensor,
                       keep_local: torch.Tensor) -> dict[str, Any]:
        deepstack = prepared.extra.get("deepstack")
        if deepstack is None:
            return {}
        # Mark kept visual tokens on the full sequence, then compact to the pruned sequence.
        visual_kept = torch.zeros_like(keep_mask)
        visual_kept[prepared.visual_positions[keep_local]] = True
        return {
            "visual_pos_masks": visual_kept[keep_mask][None],          # [1, S']
            "deepstack_visual_embeds": [d[keep_local] for d in deepstack],
        }
