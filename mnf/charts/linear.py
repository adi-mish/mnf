from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


def _center(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    return x - mean, mean.squeeze(0)


@dataclass
class LinearChart:
    """Low-rank linear chart learned by PCA/SVD.

    This is not an SAE.  It is a simple chart baseline useful for distinguishing
    'linear decodable' from 'mechanistically natural'.
    """

    components: np.ndarray  # shape: latent_dim, activation_dim
    mean: np.ndarray
    name: str = "linear_chart"

    @classmethod
    def fit(cls, activations: np.ndarray, latent_dim: int, name: str = "linear_chart") -> "LinearChart":
        x = np.asarray(activations, dtype=float)
        if x.ndim != 2:
            raise ValueError("activations must be 2D")
        if not 1 <= latent_dim <= min(x.shape):
            raise ValueError("latent_dim must be in [1, min(n,d)]")
        xc, mean = _center(x)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        return cls(components=vt[:latent_dim], mean=mean, name=name)

    def encode(self, activations: np.ndarray) -> np.ndarray:
        x = np.asarray(activations, dtype=float)
        return (x - self.mean) @ self.components.T

    def decode(self, latents: np.ndarray) -> np.ndarray:
        z = np.asarray(latents, dtype=float)
        return z @ self.components + self.mean

    def reconstruct(self, activations: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(activations))

    def intervention_add(self, activation: np.ndarray, latent_delta: np.ndarray) -> np.ndarray:
        return np.asarray(activation, dtype=float) + np.asarray(latent_delta, dtype=float) @ self.components

    def description_length(self) -> float:
        return float(self.components.size + self.mean.size) / 64.0


@dataclass
class KMeansDictionary:
    """Tiny k-means dictionary for label-free candidate atoms.

    The representation is one-hot nearest-centroid.  This intentionally serves
    as a simple contrast to sparse linear features and cyclic charts.
    """

    centroids: np.ndarray
    name: str = "kmeans_dictionary"

    @classmethod
    def fit(
        cls,
        activations: np.ndarray,
        n_atoms: int,
        n_iter: int = 30,
        seed: int = 0,
        name: str = "kmeans_dictionary",
    ) -> "KMeansDictionary":
        x = np.asarray(activations, dtype=float)
        rng = np.random.default_rng(seed)
        if n_atoms < 1 or n_atoms > len(x):
            raise ValueError("n_atoms must be in [1, n_samples]")
        centroids = x[rng.choice(len(x), size=n_atoms, replace=False)].copy()
        for _ in range(n_iter):
            dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            for j in range(n_atoms):
                if np.any(labels == j):
                    centroids[j] = x[labels == j].mean(axis=0)
                else:
                    centroids[j] = x[rng.integers(0, len(x))]
        return cls(centroids=centroids, name=name)

    def encode(self, activations: np.ndarray) -> np.ndarray:
        x = np.asarray(activations, dtype=float)
        dist = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        out = np.zeros((len(x), len(self.centroids)), dtype=float)
        out[np.arange(len(x)), labels] = 1.0
        return out

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=float) @ self.centroids

    def reconstruct(self, activations: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(activations))
