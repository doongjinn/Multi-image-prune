from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

SELECTION_MODES = ("per_image", "global", "global_min_quota")


@dataclass
class RIPruneConfig:
    num_prototypes: int = 16
    lambda_r: float = 1.0
    lambda_c: float = 1.0
    lambda_u: float = 1.0
    selection: str = "per_image"
    min_quota: int | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")


def _offsets(counts: Sequence[int]) -> list[int]:
    out = [0]
    for n in counts:
        out.append(out[-1] + int(n))
    return out


def build_prototypes(tokens: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    n = tokens.shape[0]
    P = min(num_prototypes, n)
    base, rem = divmod(n, P)
    sizes = torch.full((P,), base, dtype=torch.long, device=tokens.device)
    sizes[:rem] += 1
    ids = torch.repeat_interleave(torch.arange(P, device=tokens.device), sizes)
    sums = torch.zeros(P, tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
    sums.index_add_(0, ids, tokens)
    return sums / sizes.to(tokens.dtype).unsqueeze(1)


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


@torch.no_grad()
def ri_prune_scores(
    visual_tokens: torch.Tensor,
    image_token_counts: Sequence[int],
    query_embeds: torch.Tensor,
    config: RIPruneConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or RIPruneConfig()
    counts = [int(c) for c in image_token_counts]
    J = visual_tokens.shape[0]
    if sum(counts) != J:
        raise ValueError(f"image_token_counts sum to {sum(counts)}, expected {J}")
    device = visual_tokens.device
    offs = _offsets(counts)
    v = F.normalize(visual_tokens.float(), dim=-1)

    if query_embeds is None or query_embeds.numel() == 0:
        r = torch.zeros(J, device=device)
    else:
        q = F.normalize(query_embeds.to(device).float(), dim=-1)
        r = (v @ q.T).amax(dim=1)

    imgs = [a for a, n in enumerate(counts) if n > 0]
    protos = {a: F.normalize(build_prototypes(visual_tokens[offs[a]:offs[a + 1]].float(), cfg.num_prototypes), dim=-1)
              for a in imgs}

    c = torch.zeros(J, device=device)
    if len(imgs) > 1:
        all_p = torch.cat([protos[a] for a in imgs])
        p_owner = torch.cat([torch.full((protos[a].shape[0],), a, device=device) for a in imgs])
        t_owner = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.tensor(counts, device=device))
        sim = (v @ all_p.T).masked_fill(t_owner[:, None] == p_owner[None, :], float("-inf"))
        c = sim.amax(dim=1)

    u = torch.zeros(J, device=device)
    for a in imgs:
        sl = slice(offs[a], offs[a + 1])
        best_sim, best = (v[sl] @ protos[a].T).max(dim=1)
        n_m = torch.bincount(best, minlength=protos[a].shape[0]).float()
        u[sl] = best_sim.clamp_min(0) * torch.log(counts[a] / n_m[best])

    score = (cfg.lambda_r * zscore(r, cfg.eps) + cfg.lambda_c * zscore(c, cfg.eps)
             + cfg.lambda_u * zscore(u, cfg.eps))
    return {"score": score, "r": r, "c": c, "u": u}


def allocate_quotas(budget: int, image_token_counts: Sequence[int]) -> list[int]:
    counts = [int(c) for c in image_token_counts]
    imgs = [a for a, n in enumerate(counts) if n > 0]
    quotas = [0] * len(counts)
    if not imgs or budget <= 0:
        return quotas
    base, rem = divmod(budget, len(imgs))
    for j, a in enumerate(imgs):
        quotas[a] = base + (1 if j < rem else 0)
    return quotas


def _fill(score: torch.Tensor, chosen: torch.Tensor, n_extra: int) -> torch.Tensor:
    if n_extra <= 0:
        return chosen
    avail = score.clone()
    avail[chosen] = float("-inf")
    k = min(n_extra, int(torch.isfinite(avail).sum().item()))
    return torch.cat([chosen, avail.topk(k).indices])


def _topk_per_image(score, offs, ks):
    parts = [score[offs[a]:offs[a + 1]].topk(k).indices + offs[a] for a, k in enumerate(ks) if k > 0]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=score.device)


def select_tokens(
    score: torch.Tensor,
    image_token_counts: Sequence[int],
    budget: int,
    selection: str = "per_image",
    min_quota: int | None = None,
) -> torch.Tensor:
    counts = [int(c) for c in image_token_counts]
    J = score.shape[0]
    budget = max(0, min(int(budget), J))
    score = score.float()
    if budget == J:
        return torch.arange(J, device=score.device)
    if budget == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    offs = _offsets(counts)

    if selection == "global":
        keep = score.topk(budget).indices
    elif selection == "per_image":
        ks = [min(q, n) for q, n in zip(allocate_quotas(budget, counts), counts)]
        keep = _topk_per_image(score, offs, ks)
        keep = _fill(score, keep, budget - keep.numel())
    elif selection == "global_min_quota":
        n_img = max(1, sum(1 for n in counts if n > 0))
        mq = max(1, budget // (2 * n_img)) if min_quota is None else int(min_quota)
        keep = _topk_per_image(score, offs, [min(mq, n) for n in counts])
        if keep.numel() > budget:
            keep = keep[score[keep].topk(budget).indices]
        keep = _fill(score, keep, budget - keep.numel())
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
    cfg = config or RIPruneConfig()
    score = ri_prune_scores(visual_tokens, image_token_counts, query_embeds, cfg)["score"]
    return select_tokens(score, image_token_counts, budget, cfg.selection, cfg.min_quota)
