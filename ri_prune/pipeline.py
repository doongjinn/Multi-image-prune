"""End-to-end inference: prepare -> RI-Prune -> generate."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image

from .core import RIPruneConfig, ri_prune_scores, select_tokens
from .models.base import VLMAdapter


def resolve_budget(num_visual_tokens: int, budget: int | None = None, ratio: float | None = None) -> int:
    """Number of visual tokens to keep, from an absolute budget or a retained ratio (0, 1]."""
    if (budget is None) == (ratio is None):
        raise ValueError("pass exactly one of budget or ratio")
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        budget = max(1, round(ratio * num_visual_tokens))
    return int(min(budget, num_visual_tokens))


@dataclass
class RIPruneResult:
    text: str
    num_visual_tokens: int
    num_kept_visual_tokens: int
    image_token_counts: list[int]
    num_generated_tokens: int
    latency_prepare_sec: float
    latency_select_sec: float
    latency_generate_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RIPrune:
    """RI-Prune wrapped around a model adapter.

    Example:
        adapter = load_model("Qwen/Qwen2.5-VL-7B-Instruct")
        pruner = RIPrune(adapter)
        out = pruner.generate(images, "What changed between the two images?", ratio=0.1)
    """

    def __init__(self, adapter: VLMAdapter, config: RIPruneConfig | None = None) -> None:
        self.adapter = adapter
        self.config = config or RIPruneConfig()

    def select(self, prepared, budget: int) -> torch.Tensor:
        scores = ri_prune_scores(
            prepared.visual_tokens, prepared.image_token_counts, prepared.query_embeds, self.config,
        )
        return select_tokens(
            scores["score"], prepared.image_token_counts, budget,
            self.config.selection, self.config.min_quota,
        )

    @torch.no_grad()
    def generate(
        self,
        images: list[Image.Image],
        text: str,
        budget: int | None = None,
        ratio: float | None = None,
        max_new_tokens: int = 32,
        prune: bool = True,
    ) -> RIPruneResult:
        """Answer `text` about `images`, keeping `budget` visual tokens (or a `ratio` of them).

        prune=False runs the unpruned model through the same code path (the "Vanilla" row).
        """
        device = self.adapter.device
        _sync(device)
        t0 = time.perf_counter()
        prepared = self.adapter.prepare(images, text)
        _sync(device)
        t_prepare = time.perf_counter() - t0

        keep = None
        t_select = 0.0
        if prune:
            B = resolve_budget(prepared.num_visual_tokens, budget, ratio)
            if B < prepared.num_visual_tokens:
                t1 = time.perf_counter()
                keep = self.select(prepared, B)
                _sync(device)
                t_select = time.perf_counter() - t1

        out = self.adapter.generate(prepared, keep, max_new_tokens=max_new_tokens)
        return RIPruneResult(
            text=out.text,
            num_visual_tokens=out.num_visual_tokens,
            num_kept_visual_tokens=out.num_kept_visual_tokens,
            image_token_counts=prepared.image_token_counts,
            num_generated_tokens=out.num_generated_tokens,
            latency_prepare_sec=t_prepare,
            latency_select_sec=t_select,
            latency_generate_sec=out.latency_generate_sec,
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
