from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class LinearTransitionAtom:
    """A local linear map approximating a residual update or MLP transformation."""

    matrix: np.ndarray  # input_dim x output_dim
    bias: np.ndarray
    gate_description: str = "always_on"
    name: str = "linear_transition"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.matrix + self.bias

    def jacobian(self) -> np.ndarray:
        return self.matrix

    def description_length(self) -> float:
        return float(self.matrix.size + self.bias.size) / 64.0


def fit_linear_transition(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> LinearTransitionAtom:
    """Fit y ~= x W + b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("x and y must be 2D arrays with the same number of rows")
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    gram = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1])
    params = np.linalg.solve(gram, x_aug.T @ y)
    return LinearTransitionAtom(matrix=params[:-1], bias=params[-1])


@dataclass
class MixtureOfLinearTransforms:
    """A small MOLT-like replacement model with hard gates.

    This is deliberately simple: gates are externally supplied cluster labels.
    It is enough to test the MNF distinction between state atoms and transition
    atoms without requiring a full neural training stack.
    """

    atoms: tuple[LinearTransitionAtom, ...]
    centroids: np.ndarray

    @classmethod
    def fit_with_kmeans_gates(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        n_atoms: int,
        n_iter: int = 20,
        seed: int = 0,
    ) -> "MixtureOfLinearTransforms":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)
        centroids = x[rng.choice(len(x), size=n_atoms, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _ in range(n_iter):
            dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            for j in range(n_atoms):
                if np.any(labels == j):
                    centroids[j] = x[labels == j].mean(axis=0)
                else:
                    centroids[j] = x[rng.integers(0, len(x))]
        atoms = []
        for j in range(n_atoms):
            mask = labels == j
            if mask.sum() < x.shape[1] + 1:
                atoms.append(fit_linear_transition(x, y))
            else:
                atom = fit_linear_transition(x[mask], y[mask])
                atom.name = f"transition_{j}"
                atom.gate_description = f"nearest_centroid_{j}"
                atoms.append(atom)
        return cls(atoms=tuple(atoms), centroids=centroids)

    def gate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dist = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        labels = self.gate(x)
        out = np.zeros((len(x), self.atoms[0].bias.size), dtype=float)
        for j, atom in enumerate(self.atoms):
            mask = labels == j
            if np.any(mask):
                out[mask] = atom(x[mask])
        return out

    def description_length(self) -> float:
        return sum(a.description_length() for a in self.atoms) + self.centroids.size / 64.0
