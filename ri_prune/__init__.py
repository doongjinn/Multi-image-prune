"""RI-Prune: Relation and Information Aware Visual Token Pruning for Multi-Image LVLMs (CIKM 2026)."""
from .core import (
    RIPruneConfig,
    allocate_quotas,
    build_prototypes,
    ri_prune,
    ri_prune_scores,
    select_tokens,
    zscore,
)
from .pipeline import RIPrune, RIPruneResult, resolve_budget

__version__ = "1.0.0"

__all__ = [
    "RIPrune", "RIPruneConfig", "RIPruneResult",
    "allocate_quotas", "build_prototypes", "resolve_budget",
    "ri_prune", "ri_prune_scores", "select_tokens", "zscore",
]
