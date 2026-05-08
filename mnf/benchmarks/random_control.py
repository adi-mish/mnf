from __future__ import annotations

import numpy as np


def _labelability_score(x: np.ndarray, labels: np.ndarray) -> float:
    """Train/test linear least squares classifier accuracy for binary labels."""
    n = len(x)
    split = max(2, int(0.7 * n))
    Xtr = np.concatenate([x[:split], np.ones((split, 1))], axis=1)
    ytr = labels[:split] * 2 - 1
    w = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    Xte = np.concatenate([x[split:], np.ones((n - split, 1))], axis=1)
    pred = (Xte @ w > 0).astype(int)
    if len(pred) == 0:
        return 0.0
    return float(np.mean(pred == labels[split:]))


def _causal_use_score(x: np.ndarray, labels: np.ndarray, output: np.ndarray) -> float:
    """Does intervening along the label direction predict output change?"""
    X = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    y = labels * 2 - 1
    direction = np.linalg.lstsq(X, y, rcond=None)[0][:-1]
    if np.linalg.norm(direction) < 1e-12:
        return 0.0
    direction = direction / np.linalg.norm(direction)
    proj = x @ direction
    # output should correlate with the feature if the feature is causally used.
    corr = np.corrcoef(proj, output)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def random_vs_trained_control(n: int = 512, dim: int = 32, seed: int = 0) -> dict[str, float]:
    """Synthetic control mimicking labelable random features vs learned use.

    Both systems receive structured labels.  The 'random' representation embeds
    labels enough to be decodable, but its output is independent.  The 'trained'
    representation routes the label feature into output.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    semantic_dir = rng.normal(size=dim)
    semantic_dir /= np.linalg.norm(semantic_dir)
    nuisance = rng.normal(size=(n, dim))
    random_repr = nuisance + 2.0 * (labels * 2 - 1)[:, None] * semantic_dir[None, :]
    random_output = rng.normal(size=n)
    trained_repr = nuisance + 2.0 * (labels * 2 - 1)[:, None] * semantic_dir[None, :]
    trained_output = 1.5 * (labels * 2 - 1) + 0.2 * rng.normal(size=n)
    return {
        "random_labelability": _labelability_score(random_repr, labels),
        "trained_labelability": _labelability_score(trained_repr, labels),
        "random_causal_use": _causal_use_score(random_repr, labels, random_output),
        "trained_causal_use": _causal_use_score(trained_repr, labels, trained_output),
    }
