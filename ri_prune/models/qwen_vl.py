from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import PreparedInputs, VLMAdapter


class QwenVLAdapter(VLMAdapter):
    model_types = ("qwen2_5_vl", "qwen3_vl")

    def __init__(self, model: Any, processor: Any) -> None:
        super().__init__(model, processor)
        self.is_qwen3 = model.config.model_type == "qwen3_vl"

    @torch.no_grad()
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs:
        inner = self.model.model
        inputs = self.processor(text=[self.build_prompt(images, text)], images=images, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        grid_thw = inputs.image_grid_thw.to(self.device)
        feats = inner.get_image_features(inputs.pixel_values.to(self.device), grid_thw)
        per_image, deepstack = feats if self.is_qwen3 else (feats, None)
        embeds = self.embed(input_ids)

        position_ids, _ = inner.get_rope_index(input_ids=input_ids, image_grid_thw=grid_thw,
                                               attention_mask=inputs.attention_mask.to(self.device))
        extra: dict[str, Any] = {"max_position": int(position_ids.max().item())}
        if deepstack is not None:
            extra["deepstack"] = [d.to(embeds.dtype) for d in deepstack]
        return self._finish(input_ids, embeds, torch.cat(list(per_image)), [int(f.shape[0]) for f in per_image],
                            self.model.config.image_token_id, position_ids=position_ids, extra=extra)

    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor:
        return prepared.position_ids[:, :, keep_mask]

    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor:
        p = prepared.extra["max_position"] + 1 + step
        return torch.full((3, 1, 1), p, dtype=prepared.position_ids.dtype, device=prepared.position_ids.device)

    def prefill_kwargs(self, prepared: PreparedInputs, keep_mask: torch.Tensor, keep_local: torch.Tensor) -> dict:
        deepstack = prepared.extra.get("deepstack")
        if deepstack is None:
            return {}
        visual_kept = torch.zeros_like(keep_mask)
        visual_kept[prepared.visual_positions[keep_local]] = True
        return {"visual_pos_masks": visual_kept[keep_mask][None],
                "deepstack_visual_embeds": [d[keep_local] for d in deepstack]}
