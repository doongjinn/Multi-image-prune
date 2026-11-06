from __future__ import annotations

import torch
from PIL import Image

from .base import PreparedInputs, VLMAdapter


class LlavaOneVisionAdapter(VLMAdapter):
    model_types = ("llava_onevision",)

    @torch.no_grad()
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs:
        cfg = self.model.config
        inputs = self.processor(text=[self.build_prompt(images, text)], images=images, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        per_image = self.model.model.get_image_features(
            inputs.pixel_values.to(self.device, self.model.dtype),
            inputs.image_sizes.to(self.device),
            vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            batch_num_images=inputs.get("batch_num_images"),
        )
        return self._finish(input_ids, self.embed(input_ids), torch.cat(list(per_image)),
                            [int(f.shape[0]) for f in per_image], cfg.image_token_id)

    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor:
        return torch.arange(int(keep_mask.sum().item()), device=keep_mask.device)[None]

    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor:
        return torch.tensor([[prefill_len + step]], device=prepared.inputs_embeds.device)
