from __future__ import annotations

import numpy as np


def gradient_coupling(gradient_m: np.ndarray, gradient_n: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine coupling between two training signals."""

    g_m = np.asarray(gradient_m, dtype=float).ravel()
    g_n = np.asarray(gradient_n, dtype=float).ravel()
    if g_m.shape != g_n.shape:
        raise ValueError("gradients must have the same shape")
    denom = float(np.linalg.norm(g_m) * np.linalg.norm(g_n) + eps)
    return float(np.dot(g_m, g_n) / denom)


def pairwise_gradient_coupling(gradients: list[np.ndarray]) -> np.ndarray:
    n = len(gradients)
    out = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = gradient_coupling(gradients[i], gradients[j])
            out[i, j] = value
            out[j, i] = value
    return out
