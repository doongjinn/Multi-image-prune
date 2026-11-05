"""LLaVA-OneVision adapter (llava-hf/llava-onevision-qwen2-7b-ov-hf).

Each image becomes an AnyRes-packed block of SigLIP tokens (base view + high-res crops +
image-newline tokens), all marked by the <image> token id. The language model uses 1-D RoPE,
so after pruning the compacted sequence is re-indexed 0..S'-1.
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import PreparedInputs, VLMAdapter


class LlavaOneVisionAdapter(VLMAdapter):
    model_types = ("llava_onevision",)

    def __init__(self, model: Any, processor: Any, query_scope: str = "text") -> None:
        super().__init__(model, processor, query_scope)
        self.image_token_id = int(model.config.image_token_id)

    @property
    def language_model(self) -> Any:
        return self.model.model.language_model

    def build_prompt(self, images: list[Image.Image], text: str) -> str:
        conv = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}],
        }]
        return self.processor.apply_chat_template(conv, add_generation_prompt=True)

    @torch.no_grad()
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs:
        inner = self.model.model
        prompt = self.build_prompt(images, text)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        pixel_values = inputs.pixel_values.to(self.device, self.model.dtype)
        image_sizes = inputs.image_sizes.to(self.device)
        batch_num_images = inputs.get("batch_num_images")

        embeds = self.embed(input_ids)
        # Same call as LlavaOnevisionModel.forward, so features match unpruned inference.
        per_image = inner.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=self.model.config.vision_feature_layer,
            vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
            batch_num_images=batch_num_images,
        )
        counts = [int(f.shape[0]) for f in per_image]
        image_embeds = torch.cat(list(per_image), dim=0).to(embeds.dtype)

        visual_mask = input_ids[0] == self.image_token_id
        visual_positions = visual_mask.nonzero().flatten()
        if visual_positions.numel() != image_embeds.shape[0]:
            raise RuntimeError(
                f"{visual_positions.numel()} image tokens in the prompt but {image_embeds.shape[0]} features"
            )
        embeds[0, visual_positions] = image_embeds

        return PreparedInputs(
            input_ids=input_ids,
            inputs_embeds=embeds,
            visual_positions=visual_positions,
            image_token_counts=counts,
            query_positions=self._query_positions(input_ids[0], visual_mask, text),
        )

    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor:
        n = int(keep_mask.sum().item())
        return torch.arange(n, device=keep_mask.device)[None]

    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor:
        return torch.tensor([[prefill_len + step]], device=prepared.inputs_embeds.device)
