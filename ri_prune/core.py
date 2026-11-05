"""RI-Prune scoring and token selection (model-agnostic, pure PyTorch).

Given the projector outputs of N images, concatenated image by image,

    V = [V^(1); V^(2); ...; V^(N)],   V^(a) in R^{J_a x d},

and the embeddings of the text query Q = {q_1, ..., q_L}, RI-Prune scores every
visual token v_i (from image a) with three signals (paper Sec. 4.2):

    r(v_i) = max_l cos(v_i, q_l)                              query relevance
    c(v_i) = max_{b != a, m} cos(v_i, p_m^(b))                cross-image correspondence
    u(v_i) = s_i * log(|V^(a)| / n_{m_i}^(a))                 intra-image representativeness
             m_i = argmax_m cos(v_i, p_m^(a)),  s_i = max(0, cos(v_i, p_{m_i}^(a)))

where p_m^(a) (m = 1..P) are per-image prototypes: the means of P contiguous,
near-equal chunks of V^(a), and n_m^(a) is the number of tokens of image a whose
closest prototype is m. Each signal is z-normalised over all visual tokens of
the input and fused:

    S(v_i) = lambda_r z_r(v_i) + lambda_c z_c(v_i) + lambda_u z_u(v_i).

The budget B is split into near-equal per-image quotas B^(a) (differing by at
most one token), and each image keeps its own Top-B^(a) tokens (Sec. 4.3).
Kept tokens are returned in their original order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

SELECTION_MODES = ("per_image", "global", "global_min_quota")


@dataclass
class RIPruneConfig:
    """Hyperparameters. Defaults are the ones used for every model and benchmark in the paper."""

    num_prototypes: int = 16        # P
    lambda_r: float = 1.0           # query relevance weight
    lambda_c: float = 1.0           # cross-image correspondence weight
    lambda_u: float = 1.0           # intra-image representativeness weight
    selection: str = "per_image"    # "per_image" (paper) | "global" | "global_min_quota" (Table 6)
    min_quota: int | None = None    # only for "global_min_quota"; default B // (2N)
    eps: float = 1e-6               # z-normalisation stability constant

    def __post_init__(self) -> None:
        if self.num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        if min(self.lambda_r, self.lambda_c, self.lambda_u) < 0:
            raise ValueError("lambda weights must be non-negative")
        if self.selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}, got {self.selection!r}")


# ───────────────────────────── helpers ─────────────────────────────

def _check_counts(num_tokens: int, image_token_counts: Sequence[int]) -> list[int]:
    counts = [int(c) for c in image_token_counts]
    if any(c < 0 for c in counts):
        raise ValueError("image_token_counts must be non-negative")
    if sum(counts) != num_tokens:
        raise ValueError(
            f"image_token_counts sum to {sum(counts)} but there are {num_tokens} visual tokens"
        )
    return counts


def chunk_ids(num_tokens: int, num_prototypes: int, device: torch.device | None = None) -> torch.Tensor:
    """Chunk index of each token when splitting `num_tokens` into P contiguous, near-equal chunks.

    The first (num_tokens mod P) chunks hold one extra token. If num_tokens <= P,
    every token is its own chunk (the prototypes are the tokens themselves).
    """
    P = min(num_prototypes, num_tokens)
    base, rem = divmod(num_tokens, P)
    sizes = torch.full((P,), base, dtype=torch.long, device=device)
    sizes[:rem] += 1
    return torch.repeat_interleave(torch.arange(P, device=device), sizes)


def build_prototypes(tokens: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    """Per-image prototypes: means of P contiguous chunks of `tokens` ([J_a, d] -> [min(P, J_a), d])."""
    ids = chunk_ids(tokens.shape[0], num_prototypes, tokens.device)
    P = int(ids.max().item()) + 1
    sums = torch.zeros(P, tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
    sums.index_add_(0, ids, tokens)
    sizes = torch.bincount(ids, minlength=P).to(tokens.dtype).unsqueeze(1)
    return sums / sizes


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(x - mean) / (std + eps) over all entries; a constant signal maps to zeros."""
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


# ───────────────────────────── scoring ─────────────────────────────

@torch.no_grad()
def ri_prune_scores(
    visual_tokens: torch.Tensor,
    image_token_counts: Sequence[int],
    query_embeds: torch.Tensor,
    config: RIPruneConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Score every visual token.

    Args:
        visual_tokens: [J, d] projector outputs, concatenated image by image.
        image_token_counts: number of tokens of each image, in order (sums to J).
        query_embeds: [L, d] input embeddings of the text query tokens.
        config: hyperparameters (paper defaults if None).

    Returns:
        dict with "score" (fused S), and the raw signals "r", "c", "u", each [J] float32.
    """
    cfg = config or RIPruneConfig()
    J = visual_tokens.shape[0]
    counts = _check_counts(J, image_token_counts)
    device = visual_tokens.device

    v = F.normalize(visual_tokens.float(), dim=-1)

    # r: query relevance
    if query_embeds is None or query_embeds.numel() == 0:
        r = torch.zeros(J, device=device)
    else:
        q = F.normalize(query_embeds.to(device).float(), dim=-1)
        r = (v @ q.T).amax(dim=1)

    # prototypes per image (computed on raw tokens, then normalised for cosine)
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    protos: list[torch.Tensor | None] = []
    for a, n in enumerate(counts):
        if n == 0:
            protos.append(None)
            continue
        p = build_prototypes(visual_tokens[offsets[a]:offsets[a + 1]].float(), cfg.num_prototypes)
        protos.append(F.normalize(p, dim=-1))

    # c: cross-image correspondence (0 when there is a single non-empty image)
    c = torch.zeros(J, device=device)
    nonempty = [a for a, p in enumerate(protos) if p is not None]
    if len(nonempty) > 1:
        all_protos = torch.cat([protos[a] for a in nonempty], dim=0)                    # [sum P_a, d]
        proto_owner = torch.cat([
            torch.full((protos[a].shape[0],), a, dtype=torch.long, device=device) for a in nonempty
        ])
        token_owner = torch.repeat_interleave(
            torch.arange(len(counts), device=device), torch.tensor(counts, device=device)
        )
        sim = v @ all_protos.T                                                           # [J, sum P_a]
        sim = sim.masked_fill(token_owner[:, None] == proto_owner[None, :], float("-inf"))
        c = sim.amax(dim=1)

    # u: intra-image representativeness
    u = torch.zeros(J, device=device)
    for a in nonempty:
        sl = slice(offsets[a], offsets[a + 1])
        sim = v[sl] @ protos[a].T                                                        # [J_a, P_a]
        best_sim, best = sim.max(dim=1)
        n_m = torch.bincount(best, minlength=sim.shape[1]).float()
        idf = torch.log(counts[a] / n_m[best])                                           # n_m[best] >= 1
        u[sl] = best_sim.clamp_min(0) * idf

    score = (cfg.lambda_r * zscore(r, cfg.eps)
             + cfg.lambda_c * zscore(c, cfg.eps)
             + cfg.lambda_u * zscore(u, cfg.eps))
    return {"score": score, "r": r, "c": c, "u": u}


# ───────────────────────────── selection ─────────────────────────────

def allocate_quotas(budget: int, image_token_counts: Sequence[int]) -> list[int]:
    """Split `budget` into near-equal quotas over non-empty images (differ by at most one)."""
    counts = [int(c) for c in image_token_counts]
    nonempty = [a for a, n in enumerate(counts) if n > 0]
    quotas = [0] * len(counts)
    if not nonempty or budget <= 0:
        return quotas
    base, rem = divmod(budget, len(nonempty))
    for j, a in enumerate(nonempty):
        quotas[a] = base + (1 if j < rem else 0)
    return quotas


def _fill_globally(score: torch.Tensor, chosen: torch.Tensor, n_extra: int) -> torch.Tensor:
    """Add the `n_extra` best-scoring tokens not yet chosen."""
    if n_extra <= 0:
        return chosen
    avail = score.clone()
    avail[chosen] = float("-inf")
    n_avail = int(torch.isfinite(avail).sum().item())
    extra = avail.topk(min(n_extra, n_avail)).indices
    return torch.cat([chosen, extra])


def select_tokens(
    score: torch.Tensor,
    image_token_counts: Sequence[int],
    budget: int,
    selection: str = "per_image",
    min_quota: int | None = None,
) -> torch.Tensor:
    """Indices (into the visual-token pool, ascending) of the tokens to keep.

    selection:
        "per_image"        each image keeps its Top-B^(a); quota an image cannot use
                           (fewer tokens than its quota) is filled by global Top-K.
        "global"           global Top-B over all images.
        "global_min_quota" every image first keeps its Top-`min_quota` (default B // (2N)),
                           the rest of the budget is filled by global Top-K.
    """
    J = score.shape[0]
    counts = _check_counts(J, image_token_counts)
    budget = max(0, min(int(budget), J))
    device = score.device
    score = score.float()
    if budget == J:
        return torch.arange(J, device=device)
    if budget == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)

    if selection == "global":
        keep = score.topk(budget).indices

    elif selection == "per_image":
        parts = []
        for a, q in enumerate(allocate_quotas(budget, counts)):
            k = min(q, counts[a])
            if k > 0:
                parts.append(score[offsets[a]:offsets[a + 1]].topk(k).indices + offsets[a])
        keep = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)
        keep = _fill_globally(score, keep, budget - keep.numel())

    elif selection == "global_min_quota":
        n_img = max(1, sum(1 for n in counts if n > 0))
        mq = max(1, budget // (2 * n_img)) if min_quota is None else int(min_quota)
        parts = []
        for a, n in enumerate(counts):
            k = min(mq, n)
            if k > 0:
                parts.append(score[offsets[a]:offsets[a + 1]].topk(k).indices + offsets[a])
        keep = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)
        if keep.numel() > budget:      # min quotas alone exceed the budget: keep the best of them
            keep = keep[score[keep].topk(budget).indices]
        keep = _fill_globally(score, keep, budget - keep.numel())

    else:
        raise ValueError(f"unknown selection mode {selection!r}")

    return keep.sort().values


def ri_prune(
    visual_tokens: torch.Tensor,
    image_token_counts: Sequence[int],
    query_embeds: torch.Tensor,
    budget: int,
    config: RIPruneConfig | None = None,
) -> torch.Tensor:
    """Score and select in one call. Returns kept indices into the visual-token pool (ascending)."""
    cfg = config or RIPruneConfig()
    scores = ri_prune_scores(visual_tokens, image_token_counts, query_embeds, cfg)
    return select_tokens(scores["score"], image_token_counts, budget, cfg.selection, cfg.min_quota)
