"""CPU unit tests for RI-Prune scoring and selection.

Run with `pytest tests/` or `python tests/test_core.py`.
"""
from __future__ import annotations

import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ri_prune.core import (  # noqa: E402
    RIPruneConfig, allocate_quotas, build_prototypes, chunk_ids, ri_prune_scores, select_tokens, zscore,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.double(), b.double(), dim=0))


def reference_scores(tokens, counts, query, P, lr=1.0, lc=1.0, lu=1.0, eps=1e-6):
    """Literal, token-by-token transcription of the paper's equations (slow)."""
    J = tokens.shape[0]
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    owner = [a for a, n in enumerate(counts) for _ in range(n)]

    protos = []
    for a, n in enumerate(counts):
        toks = tokens[offsets[a]:offsets[a + 1]].double()
        k = min(P, n)
        base, rem = divmod(n, k)
        sizes = [base + (1 if m < rem else 0) for m in range(k)]
        ps, s = [], 0
        for sz in sizes:
            ps.append(toks[s:s + sz].mean(0))
            s += sz
        protos.append(ps)

    r, c, u = [], [], []
    for i in range(J):
        a = owner[i]
        v = tokens[i]
        r.append(max(_cos(v, q) for q in query))
        others = [_cos(v, p) for b in range(len(counts)) if b != a for p in protos[b]]
        c.append(max(others) if others else 0.0)
        own = [_cos(v, p) for p in protos[a]]
        m_i = max(range(len(own)), key=lambda m: own[m])
        n_m = sum(
            1 for j in range(offsets[a], offsets[a + 1])
            if max(range(len(protos[a])), key=lambda m: _cos(tokens[j], protos[a][m])) == m_i
        )
        u.append(max(0.0, own[m_i]) * math.log(counts[a] / n_m))

    def z(x):
        x = torch.tensor(x, dtype=torch.float64)
        return (x - x.mean()) / (x.std(unbiased=False) + eps)

    S = lr * z(r) + lc * z(c) + lu * z(u)
    return {"score": S, "r": torch.tensor(r), "c": torch.tensor(c), "u": torch.tensor(u)}


def test_chunks_are_contiguous_and_balanced():
    for n in [1, 5, 16, 17, 100, 729]:
        for P in [1, 4, 16, 64]:
            ids = chunk_ids(n, P)
            assert ids.numel() == n
            assert torch.all(ids[1:] >= ids[:-1])                     # contiguous, in order
            sizes = torch.bincount(ids)
            assert sizes.numel() == min(n, P)
            assert int(sizes.max() - sizes.min()) <= 1


def test_prototypes_are_chunk_means():
    x = torch.randn(10, 4)
    p = build_prototypes(x, 3)                                        # chunk sizes 4, 3, 3
    assert torch.allclose(p[0], x[0:4].mean(0))
    assert torch.allclose(p[1], x[4:7].mean(0))
    assert torch.allclose(p[2], x[7:10].mean(0))
    assert torch.allclose(build_prototypes(x, 50), x)                 # fewer tokens than P


def test_zscore():
    assert torch.equal(zscore(torch.full((7,), 3.0)), torch.zeros(7))
    z = zscore(torch.randn(1000))
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std(unbiased=False)) - 1) < 1e-4


def test_scores_match_reference():
    torch.manual_seed(0)
    for counts, P in [([12, 9, 15], 4), ([20, 20], 16), ([7, 3, 11, 5], 3), ([30], 5)]:
        d = 8
        tokens = torch.randn(sum(counts), d)
        query = torch.randn(6, d)
        got = ri_prune_scores(tokens, counts, query, RIPruneConfig(num_prototypes=P))
        ref = reference_scores(tokens, counts, query, P)
        for key in ("r", "c", "u", "score"):
            assert torch.allclose(got[key].double(), ref[key].double(), atol=1e-4), (key, counts, P)


def test_single_image_has_no_cross_image_signal():
    out = ri_prune_scores(torch.randn(40, 8), [40], torch.randn(3, 8))
    assert torch.equal(out["c"], torch.zeros(40))


def test_weights_disable_signals():
    torch.manual_seed(1)
    tokens, counts, q = torch.randn(50, 8), [25, 25], torch.randn(4, 8)
    only_r = ri_prune_scores(tokens, counts, q, RIPruneConfig(lambda_c=0, lambda_u=0))
    assert torch.allclose(only_r["score"], zscore(only_r["r"]))


def test_quotas():
    for _ in range(200):
        counts = [random.randint(0, 50) for _ in range(random.randint(1, 8))]
        B = random.randint(0, 200)
        q = allocate_quotas(B, counts)
        nonempty = [q[a] for a, n in enumerate(counts) if n > 0]
        if nonempty and B > 0:
            assert sum(q) == B
            assert max(nonempty) - min(nonempty) <= 1
        assert all(q[a] == 0 for a, n in enumerate(counts) if n == 0)


def test_selection_size_order_and_quota():
    random.seed(0)
    torch.manual_seed(0)
    for mode in ("per_image", "global", "global_min_quota"):
        for _ in range(200):
            counts = [random.randint(0, 40) for _ in range(random.randint(1, 6))]
            J = sum(counts)
            if J == 0:
                continue
            score = torch.randn(J)
            B = random.randint(0, J)
            keep = select_tokens(score, counts, B, mode)
            assert keep.numel() == B
            assert torch.equal(keep, keep.unique())                   # sorted and unique
            assert keep.numel() == 0 or (keep.min() >= 0 and keep.max() < J)


def test_per_image_topk_exact():
    score = torch.tensor([0.9, 0.1, 0.5, 0.8, 0.2, 0.3, 0.7, 0.0])
    keep = select_tokens(score, [4, 4], 4, "per_image")
    assert keep.tolist() == [0, 3, 5, 6]                           # top-2 of each image
    # global Top-4 would take three tokens from image 1
    assert select_tokens(score, [4, 4], 4, "global").tolist() == [0, 2, 3, 6]


def test_per_image_reassigns_unused_quota():
    score = torch.tensor([5.0, 4.0, 0.3, 0.2, 0.1, 0.9, 0.8])     # image 0 has 2 tokens, image 1 has 5
    keep = select_tokens(score, [2, 5], 6, "per_image")            # quotas 3 / 3 -> image 0 can use only 2
    assert keep.tolist() == [0, 1, 2, 3, 5, 6]


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
