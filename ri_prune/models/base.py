"""Common interface for plugging RI-Prune into a Hugging Face LVLM.

RI-Prune runs once, at the projector -> decoder interface:

    images, text --prepare()--> merged input embeddings (projector outputs in the image slots)
                 --RI-Prune-->  indices of visual tokens to keep
                 --generate()--> greedy decoding on the compacted sequence

`generate()` runs its own greedy loop on the language model (prefill once, then one token
at a time with a KV cache). This keeps position ids explicit for every backbone, which
`model.generate(inputs_embeds=...)` does not do reliably for M-RoPE models after tokens are
removed.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

try:
    from transformers import DynamicCache
except ImportError:  # pragma: no cover
    DynamicCache = None  # type: ignore[assignment]


@dataclass
class PreparedInputs:
    """Everything a pruner needs, computed once per sample."""

    input_ids: torch.Tensor                 # [1, S]
    inputs_embeds: torch.Tensor             # [1, S, D]  (text embeddings + projector outputs)
    visual_positions: torch.Tensor          # [J] absolute positions of visual tokens, image by image
    image_token_counts: list[int]           # tokens per image (sums to J)
    query_positions: torch.Tensor           # positions of the text tokens used as the query Q
    position_ids: torch.Tensor | None = None  # model-specific (e.g. [3, 1, S] M-RoPE ids)
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    """Start index of the last occurrence of `needle` in `haystack` (1-D), or -1."""
    n, m = haystack.numel(), needle.numel()
    if m == 0 or m > n:
        return -1
    windows = haystack.unfold(0, m, 1)                     # [n - m + 1, m]
    hits = (windows == needle.to(haystack.device)).all(dim=1).nonzero().flatten()
    return int(hits[-1].item()) if hits.numel() else -1


class VLMAdapter(ABC):
    """Model-specific glue. Subclasses implement prompt building, `prepare` and positions."""

    model_types: tuple[str, ...] = ()

    def __init__(self, model: Any, processor: Any, query_scope: str = "text") -> None:
        """
        Args:
            query_scope: which text tokens form the query Q for the relevance signal r.
                "text"   all non-visual tokens of the prompt (chat template included);
                         this is what the paper's experiments used.
                "prompt" only the tokens of the user's question (falls back to "text"
                         if the span cannot be located).
        """
        if query_scope not in ("text", "prompt"):
            raise ValueError("query_scope must be 'text' or 'prompt'")
        self.model = model
        self.processor = processor
        self.query_scope = query_scope
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.device = next(model.parameters()).device
        self.eos_token_ids = self._eos_ids()

    # ── model internals ──
    @property
    @abstractmethod
    def language_model(self) -> Any: ...

    @property
    def lm_head(self) -> Any:
        return self.model.lm_head

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for src in (getattr(self.model, "generation_config", None), self.tokenizer):
            eos = getattr(src, "eos_token_id", None)
            if eos is None:
                continue
            ids.update(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
        return ids

    # ── to implement per model ──
    @abstractmethod
    def build_prompt(self, images: list[Image.Image], text: str) -> str: ...

    @abstractmethod
    def prepare(self, images: list[Image.Image], text: str) -> PreparedInputs: ...

    @abstractmethod
    def prefill_position_ids(self, prepared: PreparedInputs, keep_mask: torch.Tensor) -> torch.Tensor | None:
        """Position ids for the compacted prompt (keep_mask: [S] bool over the full sequence)."""

    @abstractmethod
    def decode_position_ids(self, prepared: PreparedInputs, prefill_len: int, step: int) -> torch.Tensor | None:
        """Position ids of the `step`-th generated token (0-based)."""

    def prefill_kwargs(self, prepared: PreparedInputs, keep_mask: torch.Tensor,
                       keep_local: torch.Tensor) -> dict[str, Any]:
        """Extra kwargs for the language model's prefill call (e.g. Qwen3-VL deepstack)."""
        return {}

    # ── shared helpers ──
    def _query_positions(self, input_ids: torch.Tensor, visual_mask: torch.Tensor, text: str) -> torch.Tensor:
        text_pos = (~visual_mask).nonzero().flatten()
        if self.query_scope == "text":
            return text_pos
        q_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        start = find_subsequence(input_ids, q_ids)
        if start < 0:
            return text_pos
        return torch.arange(start, start + q_ids.numel(), device=input_ids.device)

    def keep_mask(self, prepared: PreparedInputs, keep_local: torch.Tensor | None) -> torch.Tensor:
        """[S] bool mask over the full sequence: all text tokens plus kept visual tokens."""
        S = prepared.input_ids.shape[1]
        mask = torch.ones(S, dtype=torch.bool, device=prepared.inputs_embeds.device)
        if keep_local is not None:
            mask[prepared.visual_positions] = False
            mask[prepared.visual_positions[keep_local.to(mask.device)]] = True
        return mask

    @torch.no_grad()
    def generate(
        self,
        prepared: PreparedInputs,
        keep_local: torch.Tensor | None = None,
        max_new_tokens: int = 32,
    ) -> GenerationOutput:
        """Greedy decoding after dropping visual tokens not in `keep_local` (None = keep all)."""
        device = prepared.inputs_embeds.device
        if keep_local is not None:
            keep_local = keep_local.to(device).long().sort().values
        mask = self.keep_mask(prepared, keep_local)
        local = keep_local if keep_local is not None else torch.arange(prepared.num_visual_tokens, device=device)

        embeds = prepared.inputs_embeds[:, mask]
        prefill_len = embeds.shape[1]
        pos = self.prefill_position_ids(prepared, mask)
        extra = self.prefill_kwargs(prepared, mask, local)

        _sync(device)
        t0 = time.perf_counter()
        cache = DynamicCache() if DynamicCache is not None else None
        attn = torch.ones(1, prefill_len, dtype=torch.long, device=device)
        out = self.language_model(
            inputs_embeds=embeds, position_ids=pos, attention_mask=attn,
            past_key_values=cache, use_cache=True, return_dict=True, **extra,
        )
        cache = out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1])

        generated: list[int] = []
        for step in range(max_new_tokens):
            nxt = logits.argmax(dim=-1)                                  # [1]
            tok = int(nxt.item())
            if tok in self.eos_token_ids:
                break
            generated.append(tok)
            if step == max_new_tokens - 1:
                break
            attn = torch.ones(1, prefill_len + step + 1, dtype=torch.long, device=device)
            out = self.language_model(
                inputs_embeds=self.embed(nxt[:, None]),
                position_ids=self.decode_position_ids(prepared, prefill_len, step),
                attention_mask=attn, past_key_values=cache, use_cache=True, return_dict=True,
            )
            cache = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1])
        _sync(device)
        latency = time.perf_counter() - t0

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationOutput(
            text=text,
            num_visual_tokens=prepared.num_visual_tokens,
            num_kept_visual_tokens=int(local.numel()),
            prefill_length=prefill_len,
            num_generated_tokens=len(generated),
            latency_generate_sec=latency,
        )
