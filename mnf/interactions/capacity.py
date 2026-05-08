from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def capacity_competition_from_matrices(
    membership_m: Sequence[float],
    membership_n: Sequence[float],
    coactivation: np.ndarray,
    gram_squared: np.ndarray,
) -> float:
    """Mechanism-level superposition/interference term.

    Computes sum_ij pi_m[i] pi_n[j] q_ij <d_i, d_j>^2.
    """

    pi_m = np.asarray(membership_m, dtype=float)
    pi_n = np.asarray(membership_n, dtype=float)
    q = np.asarray(coactivation, dtype=float)
    gram2 = np.asarray(gram_squared, dtype=float)
    if q.shape != gram2.shape or q.shape != (pi_m.size, pi_n.size):
        raise ValueError("coactivation and gram_squared must match membership sizes")
    return float(np.sum(pi_m[:, None] * pi_n[None, :] * q * gram2))


def capacity_competition(
    membership_m: Mapping[str, float],
    membership_n: Mapping[str, float],
    coactivation: Mapping[tuple[str, str], float],
    gram_squared: Mapping[tuple[str, str], float],
) -> float:
    total = 0.0
    for i, pi_i in membership_m.items():
        for j, pi_j in membership_n.items():
            q = float(coactivation.get((i, j), coactivation.get((j, i), 0.0)))
            g2 = float(gram_squared.get((i, j), gram_squared.get((j, i), 0.0)))
            total += max(float(pi_i), 0.0) * max(float(pi_j), 0.0) * q * g2
    return float(total)


def decoder_gram_squared(decoder_directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(decoder_directions, dtype=float)
    if directions.ndim != 2:
        raise ValueError("decoder_directions must be a 2D array")
    norms = np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    normalized = directions / norms
    return (normalized.T @ normalized) ** 2
