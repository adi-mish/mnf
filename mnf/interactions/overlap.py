from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _as_membership_map(values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(values, Mapping):
        return {str(k): float(v) for k, v in values.items()}
    return {str(i): float(v) for i, v in enumerate(values)}


def soft_jaccard(
    membership_a: Mapping[str, float] | Sequence[float],
    membership_b: Mapping[str, float] | Sequence[float],
) -> float:
    """Soft atom overlap using sum(min) / sum(max)."""

    a = _as_membership_map(membership_a)
    b = _as_membership_map(membership_b)
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    numerator = sum(min(max(a.get(k, 0.0), 0.0), max(b.get(k, 0.0), 0.0)) for k in keys)
    denominator = sum(max(max(a.get(k, 0.0), 0.0), max(b.get(k, 0.0), 0.0)) for k in keys)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def weighted_membership_dot(
    membership_a: Mapping[str, float] | Sequence[float],
    membership_b: Mapping[str, float] | Sequence[float],
) -> float:
    """Simple weighted overlap without Jaccard normalization."""

    a = _as_membership_map(membership_a)
    b = _as_membership_map(membership_b)
    keys = set(a) | set(b)
    return float(sum(max(a.get(k, 0.0), 0.0) * max(b.get(k, 0.0), 0.0) for k in keys))


def pairwise_overlap_matrix(memberships: Sequence[Mapping[str, float] | Sequence[float]]) -> np.ndarray:
    n = len(memberships)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            out[i, j] = soft_jaccard(memberships[i], memberships[j])
    return out
