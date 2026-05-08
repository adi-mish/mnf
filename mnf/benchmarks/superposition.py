from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def normalize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=0, keepdims=True) + eps
    return x / norms


@dataclass
class SparseFeatureWorld:
    """Toy sparse feature world for superposition/rate-distortion experiments."""

    n_features: int
    activation_dim: int
    feature_probs: np.ndarray
    dictionary: np.ndarray  # activation_dim x n_features
    importance: np.ndarray
    seed: int = 0

    @classmethod
    def random(
        cls,
        n_features: int = 32,
        activation_dim: int = 8,
        feature_prob: float | np.ndarray = 0.05,
        seed: int = 0,
    ) -> "SparseFeatureWorld":
        rng = np.random.default_rng(seed)
        if np.isscalar(feature_prob):
            probs = np.full(n_features, float(feature_prob))
        else:
            probs = np.asarray(feature_prob, dtype=float)
            if probs.shape != (n_features,):
                raise ValueError("feature_prob array must have shape (n_features,)")
        dictionary = normalize_columns(rng.normal(size=(activation_dim, n_features)))
        importance = 0.5 + rng.random(n_features)
        return cls(n_features, activation_dim, probs, dictionary, importance, seed)

    def sample_latents(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        active = rng.random((n, self.n_features)) < self.feature_probs[None, :]
        magnitudes = rng.exponential(scale=1.0, size=(n, self.n_features))
        return active * magnitudes

    def activations(self, latents: np.ndarray, noise: float = 0.0, seed: int | None = None) -> np.ndarray:
        x = np.asarray(latents, dtype=float) @ self.dictionary.T
        if noise > 0:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            x = x + noise * rng.normal(size=x.shape)
        return x

    def interference_matrix(self) -> np.ndarray:
        gram = self.dictionary.T @ self.dictionary
        probs = self.feature_probs[:, None] * self.feature_probs[None, :]
        return probs * gram**2

    def total_interference(self) -> float:
        mat = self.interference_matrix()
        return float((mat.sum() - np.trace(mat)) / 2.0)

    def packing_ratio(self) -> float:
        return self.n_features / self.activation_dim

    def decode_by_pseudoinverse(self, activations: np.ndarray) -> np.ndarray:
        return np.asarray(activations, dtype=float) @ np.linalg.pinv(self.dictionary.T)


def predicted_superposition_favored(
    n_features: int,
    activation_dim: int,
    mean_coactivation: float,
    mean_importance: float = 1.0,
    interference_weight: float = 1.0,
) -> bool:
    """A minimal phase-boundary heuristic from the sparse-packing theorem.

    Superposition is favored when the value of storing extra features exceeds
    expected interference.  This is a simple operationalization for experiments,
    not the final theorem.
    """
    extra = max(0, n_features - activation_dim)
    if extra == 0:
        return False
    value = extra * mean_importance
    interference = interference_weight * extra * mean_coactivation * n_features / max(1, activation_dim)
    return bool(value > interference)


def superposition_phase_statistic(world: SparseFeatureWorld) -> dict[str, float]:
    coh = world.dictionary.T @ world.dictionary
    off_diag = coh[~np.eye(world.n_features, dtype=bool)]
    return {
        "packing_ratio": world.packing_ratio(),
        "mean_abs_coherence": float(np.mean(np.abs(off_diag))),
        "max_abs_coherence": float(np.max(np.abs(off_diag))),
        "total_interference": world.total_interference(),
        "mean_coactivation": float(np.mean(world.feature_probs[:, None] * world.feature_probs[None, :])),
    }
